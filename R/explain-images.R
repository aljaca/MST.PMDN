################################################################################
# Functional whole-image contrasts and spatial occlusion                       #
################################################################################

.image_shape_mst_pmdn <- function(images, name = "image_inputs") {
  size <- if (inherits(images, "torch_tensor")) {
    as.integer(images$size())
  } else {
    dim(images)
  }
  if (is.null(size) || length(size) != 4L || any(size < 1L)) {
    stop(
      sprintf("%s must have shape [case, channel, row, column].", name),
      call. = FALSE
    )
  }
  as.integer(size)
}

.broadcast_reference_images_mst_pmdn <- function(reference_images,
                                                   image_shape) {
  reference_shape <- .image_shape_mst_pmdn(
    reference_images, "reference_images"
  )
  if (!identical(reference_shape[2:4], image_shape[2:4]) ||
      !reference_shape[1] %in% c(1L, image_shape[1])) {
    stop(
      paste0(
        "reference_images must match image channels and spatial dimensions, ",
        "with either one reference row or one row per case."
      ),
      call. = FALSE
    )
  }
  if (reference_shape[1] == image_shape[1]) return(reference_images)
  if (inherits(reference_images, "torch_tensor")) {
    return(reference_images$expand(c(image_shape[1], -1L, -1L, -1L)))
  }
  reference_images[rep(1L, image_shape[1]), , , , drop = FALSE]
}

.coerce_reference_images_like_mst_pmdn <- function(reference_images,
                                                     image_inputs) {
  if (inherits(image_inputs, "torch_tensor")) {
    if (!inherits(reference_images, "torch_tensor")) {
      return(torch_tensor(
        reference_images,
        dtype = image_inputs$dtype,
        device = image_inputs$device
      ))
    }
    return(reference_images$to(
      dtype = image_inputs$dtype,
      device = image_inputs$device
    ))
  }
  if (inherits(reference_images, "torch_tensor")) {
    return(torch::as_array(reference_images$to(device = "cpu")))
  }
  reference_images
}

.validate_rebuilt_compatibility_mst_pmdn <- function(candidate,
                                                       original,
                                                       name) {
  candidate_shape <- .image_shape_mst_pmdn(candidate, name)
  original_shape <- .image_shape_mst_pmdn(original, "rebuilt original images")
  if (!identical(candidate_shape, original_shape)) {
    stop(
      sprintf("%s must match the rebuilt original image shape.", name),
      call. = FALSE
    )
  }
  candidate_tensor <- inherits(candidate, "torch_tensor")
  original_tensor <- inherits(original, "torch_tensor")
  if (!identical(candidate_tensor, original_tensor)) {
    stop(
      sprintf(
        "%s and rebuilt original images must use the same representation.",
        name
      ),
      call. = FALSE
    )
  }
  same_device <- !candidate_tensor || (
    identical(candidate$device$type, original$device$type) &&
      identical(candidate$device$index, original$device$index)
  )
  if (candidate_tensor &&
      (candidate$dtype != original$dtype || !same_device)) {
    stop(
      sprintf(
        "%s must match the rebuilt original image dtype and device.",
        name
      ),
      call. = FALSE
    )
  }
  invisible(TRUE)
}

.validate_cases_mst_pmdn <- function(cases, n) {
  if (is.null(cases)) return(seq_len(n))
  if (!is.numeric(cases) || !length(cases) || anyNA(cases) ||
      any(cases < 1L) || any(cases > n) || any(cases != floor(cases))) {
    stop("cases must contain valid 1-based case indices.", call. = FALSE)
  }
  unique(as.integer(cases))
}

.mask_like_images_mst_pmdn <- function(images,
                                        mask_2d,
                                        n,
                                        channels = NULL) {
  shape <- .image_shape_mst_pmdn(images)
  if (!is.matrix(mask_2d) || !identical(dim(mask_2d), shape[3:4])) {
    stop("mask_2d must match the image spatial dimensions.", call. = FALSE)
  }
  if (is.null(channels)) {
    mask <- array(
      rep(mask_2d, n),
      dim = c(shape[3], shape[4], n)
    )
    mask <- aperm(mask, c(3L, 1L, 2L))
    dim(mask) <- c(n, 1L, shape[3], shape[4])
  } else {
    mask <- array(0, dim = c(n, shape[2], shape[3], shape[4]))
    for (channel in channels) {
      mask[, channel, , ] <- aperm(
        array(
          rep(mask_2d, n), dim = c(shape[3], shape[4], n)
        ),
        c(3L, 1L, 2L)
      )
    }
  }
  if (inherits(images, "torch_tensor")) {
    torch_tensor(mask, dtype = images$dtype, device = images$device)
  } else {
    mask
  }
}

.rebuild_or_blend_images_mst_pmdn <- function(base_images,
                                               reference_images,
                                               masks,
                                               rebuild_channels,
                                               case_index) {
  if (is.null(rebuild_channels)) {
    if (!inherits(base_images, "torch_tensor") &&
        dim(masks)[2L] == 1L && dim(base_images)[2L] > 1L) {
      masks <- masks[, rep(1L, dim(base_images)[2L]), , , drop = FALSE]
    }
    return((1 - masks) * base_images + masks * reference_images)
  }
  if (!is.function(rebuild_channels)) {
    stop("rebuild_channels must be NULL or a function.", call. = FALSE)
  }
  rebuilt <- rebuild_channels(
    base_images = base_images,
    reference_images = reference_images,
    masks = masks,
    case_index = case_index
  )
  shape <- .image_shape_mst_pmdn(rebuilt, "rebuild_channels result")
  if (shape[1] != length(case_index)) {
    stop("rebuild_channels must return one model-ready image per case.",
         call. = FALSE)
  }
  rebuilt
}

.zero_mask_mst_pmdn <- function(images, n, channels = NULL) {
  shape <- .image_shape_mst_pmdn(images)
  .mask_like_images_mst_pmdn(
    images,
    matrix(0, nrow = shape[3], ncol = shape[4]),
    n = n,
    channels = channels
  )
}

.one_mask_mst_pmdn <- function(images, n, channels = NULL) {
  shape <- .image_shape_mst_pmdn(images)
  .mask_like_images_mst_pmdn(
    images,
    matrix(1, nrow = shape[3], ncol = shape[4]),
    n = n,
    channels = channels
  )
}

.normalize_latent_banks_mst_pmdn <- function(pred,
                                               functional,
                                               latent_draws,
                                               num_samples,
                                               seed,
                                               device) {
  if (!.functional_is_monte_carlo_mst_pmdn(functional$type)) {
    return(list(NULL))
  }
  banks <- if (is.null(latent_draws) ||
               inherits(latent_draws, "mst_pmdn_latent_draws")) {
    list(latent_draws)
  } else if (is.list(latent_draws) && length(latent_draws) &&
             all(vapply(
               latent_draws,
               inherits,
               logical(1),
               what = "mst_pmdn_latent_draws"
             ))) {
    latent_draws
  } else {
    stop(
      paste0(
        "latent_draws must be a latent bank or a non-empty list of latent ",
        "banks for image occlusion."
      ),
      call. = FALSE
    )
  }
  banks <- lapply(banks, function(bank) {
    .ensure_latent_bank_mst_pmdn(
      pred, functional, bank, num_samples, seed, device
    )
  })
  bank_samples <- vapply(banks, function(bank) {
    bank$num_samples
  }, integer(1))
  if (length(unique(bank_samples)) != 1L) {
    stop("Replicate latent banks must contain the same number of draws.",
         call. = FALSE)
  }
  banks
}

.latent_banks_for_output_mst_pmdn <- function(banks) {
  banks <- lapply(banks, .latent_draws_for_output_mst_pmdn)
  if (length(banks) == 1L) banks[[1L]] else banks
}

.mc_effect_summary_mst_pmdn <- function(values) {
  values <- as.matrix(values)
  replicate_count <- ncol(values)
  minimum <- apply(values, 1L, min)
  maximum <- apply(values, 1L, max)
  list(
    effect = rowMeans(values),
    mc_effect_sd = if (replicate_count > 1L) {
      apply(values, 1L, stats::sd)
    } else {
      rep(NA_real_, nrow(values))
    },
    mc_effect_min = minimum,
    mc_effect_max = maximum,
    mc_sign_stable = if (replicate_count > 1L) {
      (minimum > 0 & maximum > 0) | (minimum < 0 & maximum < 0)
    } else {
      rep(NA, nrow(values))
    }
  )
}

.normalize_channel_groups_mst_pmdn <- function(channel_groups, n_channels) {
  if (is.null(channel_groups)) {
    return(list(all = NULL))
  }
  if (is.numeric(channel_groups)) channel_groups <- list(channel_groups)
  if (!is.list(channel_groups) || !length(channel_groups)) {
    stop("channel_groups must be NULL, an index vector, or a named list.",
         call. = FALSE)
  }
  channel_groups <- lapply(channel_groups, function(group) {
    if (!is.numeric(group) || !length(group) || anyNA(group) ||
        any(group < 1L) || any(group > n_channels) ||
        any(group != floor(group))) {
      stop("Every channel group must contain valid 1-based channel indices.",
           call. = FALSE)
    }
    unique(as.integer(group))
  })
  if (is.null(names(channel_groups))) {
    names(channel_groups) <- paste0("group_", seq_along(channel_groups))
  }
  empty_names <- !nzchar(names(channel_groups))
  names(channel_groups)[empty_names] <- paste0(
    "group_", which(empty_names)
  )
  if (anyDuplicated(names(channel_groups))) {
    stop("channel_groups must have unique names.", call. = FALSE)
  }
  channel_groups
}

# Contrast observed and reference image fields for a predictive functional
image_contrast_mst_pmdn <- function(model,
                                    inputs,
                                    image_inputs,
                                    reference_images,
                                    functional,
                                    cases = NULL,
                                    rebuild_channels = NULL,
                                    decompose = FALSE,
                                    channels = c(
                                      "location", "scale", "skewness", "df"
                                    ),
                                    num_samples = 4096L,
                                    latent_draws = NULL,
                                    seed = NULL,
                                    chunk_size = NULL,
                                    device = "cpu",
                                    response_names = NULL,
                                    min_tail_draws = 20L) {
  if (!inherits(functional, "mst_functional")) {
    stop("functional must be returned by mst_functional().", call. = FALSE)
  }
  inputs_matrix <- .as_input_matrix_mst_pmdn(inputs)
  image_shape <- .image_shape_mst_pmdn(image_inputs)
  if (image_shape[1] != nrow(inputs_matrix)) {
    stop("image_inputs must be case-matched to inputs.", call. = FALSE)
  }
  reference_images <- .broadcast_reference_images_mst_pmdn(
    reference_images, image_shape
  )
  reference_images <- .coerce_reference_images_like_mst_pmdn(
    reference_images, image_inputs
  )
  cases <- .validate_cases_mst_pmdn(cases, nrow(inputs_matrix))
  case_inputs <- inputs_matrix[cases, , drop = FALSE]
  base <- .subset_rows_mst_pmdn(image_inputs, cases)
  reference <- .subset_rows_mst_pmdn(reference_images, cases)
  n <- length(cases)

  original_images <- .rebuild_or_blend_images_mst_pmdn(
    base,
    reference,
    .zero_mask_mst_pmdn(base, n),
    rebuild_channels,
    cases
  )
  reference_model_images <- .rebuild_or_blend_images_mst_pmdn(
    base,
    reference,
    .one_mask_mst_pmdn(base, n),
    rebuild_channels,
    cases
  )
  .validate_rebuilt_compatibility_mst_pmdn(
    reference_model_images,
    original_images,
    "rebuilt reference images"
  )
  pred_original <- .predict_chunks_mst_pmdn(
    model, case_inputs, original_images,
    chunk_size = chunk_size, device = device
  )
  if (isTRUE(decompose)) {
    .require_single_component_mst_pmdn(pred_original)
  }
  pred_reference <- .predict_chunks_mst_pmdn(
    model, case_inputs, reference_model_images,
    chunk_size = chunk_size, device = device
  )
  num_samples <- validate_num_samples(num_samples)
  min_tail_draws <- validate_num_samples(min_tail_draws)
  latent_draws <- .ensure_latent_bank_mst_pmdn(
    pred_original, functional, latent_draws, num_samples, seed, device
  )
  if (!is.null(latent_draws)) num_samples <- latent_draws$num_samples

  active_channels <- character(0)
  if (isTRUE(decompose)) {
    decomposition <- .muffle_tail_resolution_warnings_mst_pmdn(
      decompose_mst_pmdn(
      pred_from = pred_reference,
      pred_to = pred_original,
      functional = functional,
      channels = channels,
      latent_draws = latent_draws,
      num_samples = num_samples,
      chunk_size = chunk_size,
      device = device,
      response_names = response_names,
      min_tail_draws = min_tail_draws
    ))
    data <- data.frame(
      case = cases,
      reference = decomposition$data$from,
      original = decomposition$data$to,
      contrast = decomposition$data$total
    )
    active_channels <- decomposition$active_channels
    for (channel in active_channels) {
      data[[paste0("channel_", channel)]] <-
        decomposition$data[[paste0("channel_", channel)]]
    }
    data$sum_to_total_residual <-
      decomposition$data$sum_to_total_residual
    diagnostics <- decomposition$diagnostics
  } else {
    original_result <- .functional_values_quiet_mst_pmdn(
      pred_original, functional, num_samples, latent_draws,
      chunk_size, device, response_names, min_tail_draws
    )
    reference_result <- .functional_values_quiet_mst_pmdn(
      pred_reference, functional, num_samples, latent_draws,
      chunk_size, device, response_names, min_tail_draws
    )
    data <- data.frame(
      case = cases,
      reference = reference_result$data$value,
      original = original_result$data$value,
      contrast = original_result$data$value - reference_result$data$value,
      sum_to_total_residual = NA_real_
    )
    tail_summary <- .tail_resolution_summary_mst_pmdn(
      list(original_result$diagnostics, reference_result$diagnostics),
      min_tail_draws
    )
    diagnostics <- c(
      list(
        original = original_result$diagnostics,
        reference = reference_result$diagnostics
      ),
      tail_summary
    )
  }
  if (isTRUE(decompose)) {
    tail_summary <- .tail_resolution_summary_mst_pmdn(
      list(diagnostics),
      min_tail_draws
    )
  }
  .warn_tail_resolution_mst_pmdn(
    tail_summary, "image_contrast_mst_pmdn()"
  )
  out <- list(
    data = data,
    functional = functional,
    cases = cases,
    active_channels = active_channels,
    reference = list(rebuild_channels = !is.null(rebuild_channels)),
    settings = list(
      decomposed = isTRUE(decompose),
      num_samples = if (is.null(latent_draws)) NA_integer_ else
        latent_draws$num_samples,
      chunk_size = chunk_size,
      device = device,
      min_tail_draws = min_tail_draws
    ),
    diagnostics = diagnostics,
    latent_draws = .latent_draws_for_output_mst_pmdn(latent_draws)
  )
  class(out) <- "mst_pmdn_image_contrast"
  out
}

.validate_pair_mst_pmdn <- function(x, name) {
  if (!is.numeric(x) || length(x) != 2L || any(!is.finite(x)) ||
      any(x < 1L) || any(x != floor(x))) {
    stop(sprintf("%s must contain two positive integers.", name),
         call. = FALSE)
  }
  as.integer(x)
}

.patch_starts_mst_pmdn <- function(size, patch, stride) {
  if (patch > size) {
    stop("patch_size cannot exceed an image spatial dimension.",
         call. = FALSE)
  }
  last <- size - patch + 1L
  unique(c(seq.int(1L, last, by = stride), last))
}

.patch_mask_mst_pmdn <- function(n_row,
                                  n_col,
                                  row_start,
                                  col_start,
                                  patch_size,
                                  taper) {
  if (taper == "cosine") {
    row_window <- sin(pi * (seq_len(patch_size[1]) - 0.5) / patch_size[1])
    col_window <- sin(pi * (seq_len(patch_size[2]) - 0.5) / patch_size[2])
    window <- outer(row_window, col_window)
  } else {
    window <- matrix(1, nrow = patch_size[1], ncol = patch_size[2])
  }
  mask <- matrix(0, nrow = n_row, ncol = n_col)
  row_index <- row_start:(row_start + patch_size[1] - 1L)
  col_index <- col_start:(col_start + patch_size[2] - 1L)
  mask[row_index, col_index] <- window
  mask
}

# Spatial occlusion effects for an MST-PMDN predictive functional
image_occlusion_mst_pmdn <- function(model,
                                     inputs,
                                     image_inputs,
                                     reference_images,
                                     functional,
                                     patch_size,
                                     stride,
                                     cases = NULL,
                                     taper = c("cosine", "none"),
                                     channel_groups = NULL,
                                     rebuild_channels = NULL,
                                     decompose = FALSE,
                                     channels = c(
                                       "location", "scale", "skewness", "df"
                                     ),
                                     num_samples = 4096L,
                                     latent_draws = NULL,
                                     seed = NULL,
                                     chunk_size = NULL,
                                     device = "cpu",
                                     response_names = NULL,
                                     min_tail_draws = 20L) {
  if (!inherits(functional, "mst_functional")) {
    stop("functional must be returned by mst_functional().", call. = FALSE)
  }
  taper <- match.arg(taper)
  patch_size <- .validate_pair_mst_pmdn(patch_size, "patch_size")
  stride <- .validate_pair_mst_pmdn(stride, "stride")
  inputs_matrix <- .as_input_matrix_mst_pmdn(inputs)
  image_shape <- .image_shape_mst_pmdn(image_inputs)
  if (image_shape[1] != nrow(inputs_matrix)) {
    stop("image_inputs must be case-matched to inputs.", call. = FALSE)
  }
  reference_images <- .broadcast_reference_images_mst_pmdn(
    reference_images, image_shape
  )
  reference_images <- .coerce_reference_images_like_mst_pmdn(
    reference_images, image_inputs
  )
  grouped_channels <- !is.null(channel_groups)
  groups <- .normalize_channel_groups_mst_pmdn(
    channel_groups, image_shape[2]
  )
  requested_channels <- if (isTRUE(decompose)) {
    .validate_channels_mst_pmdn(channels)
  } else {
    character(0)
  }
  cases <- .validate_cases_mst_pmdn(cases, nrow(inputs_matrix))
  case_inputs <- inputs_matrix[cases, , drop = FALSE]
  base <- .subset_rows_mst_pmdn(image_inputs, cases)
  reference <- .subset_rows_mst_pmdn(reference_images, cases)
  n <- length(cases)
  num_samples <- validate_num_samples(num_samples)
  min_tail_draws <- validate_num_samples(min_tail_draws)

  original_images <- .rebuild_or_blend_images_mst_pmdn(
    base,
    reference,
    .zero_mask_mst_pmdn(
      base,
      n,
      channels = if (grouped_channels) seq_len(image_shape[2]) else NULL
    ),
    rebuild_channels,
    cases
  )
  pred_original <- .predict_chunks_mst_pmdn(
    model, case_inputs, original_images,
    chunk_size = chunk_size, device = device
  )
  if (isTRUE(decompose)) {
    .require_single_component_mst_pmdn(pred_original)
  }
  latent_banks <- .normalize_latent_banks_mst_pmdn(
    pred_original,
    functional,
    latent_draws,
    num_samples,
    seed,
    device
  )
  replicate_count <- length(latent_banks)
  if (!is.null(latent_banks[[1L]])) {
    num_samples <- latent_banks[[1L]]$num_samples
  }
  original_results <- lapply(latent_banks, function(bank) {
    .functional_values_quiet_mst_pmdn(
      pred_original, functional, num_samples, bank,
      chunk_size, device, response_names, min_tail_draws
    )
  })

  row_starts <- .patch_starts_mst_pmdn(
    image_shape[3], patch_size[1], stride[1]
  )
  col_starts <- .patch_starts_mst_pmdn(
    image_shape[4], patch_size[2], stride[2]
  )
  patch_table <- expand.grid(
    row_start = row_starts,
    col_start = col_starts,
    KEEP.OUT.ATTRS = FALSE
  )
  patch_table$patch <- seq_len(nrow(patch_table))
  patch_table$row_end <- patch_table$row_start + patch_size[1] - 1L
  patch_table$col_end <- patch_table$col_start + patch_size[2] - 1L
  patch_table$row_center <- (patch_table$row_start + patch_table$row_end) / 2
  patch_table$col_center <- (patch_table$col_start + patch_table$col_end) / 2
  patch_table$mask_sum <- NA_real_
  patch_table$mask_mean <- NA_real_
  patch_table$mask_max <- NA_real_
  coverage <- matrix(0, nrow = image_shape[3], ncol = image_shape[4])
  weighted_coverage <- matrix(
    0, nrow = image_shape[3], ncol = image_shape[4]
  )
  rows_out <- list()
  replicate_rows_out <- list()
  active_channels <- character(0)
  diagnostics <- list()
  output_index <- 0L

  for (patch_row in seq_len(nrow(patch_table))) {
    patch_mask <- .patch_mask_mst_pmdn(
      image_shape[3], image_shape[4],
      patch_table$row_start[patch_row],
      patch_table$col_start[patch_row],
      patch_size,
      taper
    )
    patch_weights <- patch_mask[patch_mask > 0]
    patch_table$mask_sum[patch_row] <- sum(patch_weights)
    patch_table$mask_mean[patch_row] <- mean(patch_weights)
    patch_table$mask_max[patch_row] <- max(patch_weights)
    coverage <- coverage + (patch_mask > 0)
    weighted_coverage <- weighted_coverage + patch_mask

    for (group_name in names(groups)) {
      group <- groups[[group_name]]
      masks <- .mask_like_images_mst_pmdn(
        base, patch_mask, n = n, channels = group
      )
      occluded_images <- .rebuild_or_blend_images_mst_pmdn(
        base, reference, masks, rebuild_channels, cases
      )
      .validate_rebuilt_compatibility_mst_pmdn(
        occluded_images,
        original_images,
        "rebuilt occluded images"
      )
      # The model prediction is independent of the latent bank and is retained
      # for every replicate evaluation below.
      pred_occluded <- .predict_chunks_mst_pmdn(
        model, case_inputs, occluded_images,
        chunk_size = chunk_size, device = device
      )

      channel_matrices <- list()
      if (isTRUE(decompose)) {
        replicate_results <- lapply(latent_banks, function(bank) {
          .muffle_tail_resolution_warnings_mst_pmdn(
            decompose_mst_pmdn(
              pred_from = pred_occluded,
              pred_to = pred_original,
              functional = functional,
              channels = channels,
              latent_draws = bank,
              num_samples = num_samples,
              chunk_size = chunk_size,
              device = device,
              response_names = response_names,
              min_tail_draws = min_tail_draws
            )
          )
        })
        effect_values <- do.call(cbind, lapply(
          replicate_results, function(result) result$data$total
        ))
        patch_active <- Reduce(
          union,
          lapply(replicate_results, function(result) {
            result$active_channels
          }),
          init = character(0)
        )
        active_channels <- union(active_channels, patch_active)
        for (channel in requested_channels) {
          column <- paste0("channel_", channel)
          channel_matrices[[channel]] <- do.call(cbind, lapply(
            replicate_results,
            function(result) {
              if (channel %in% result$active_channels) {
                result$data[[column]]
              } else {
                rep(0, n)
              }
            }
          ))
        }
        replicate_diagnostics <- lapply(
          replicate_results, function(result) result$diagnostics
        )
      } else {
        replicate_results <- lapply(
          seq_along(latent_banks),
          function(replicate) {
            .functional_values_quiet_mst_pmdn(
              pred_occluded,
              functional,
              num_samples,
              latent_banks[[replicate]],
              chunk_size,
              device,
              response_names,
              min_tail_draws
            )
          }
        )
        effect_values <- do.call(cbind, lapply(
          seq_along(replicate_results),
          function(replicate) {
            original_results[[replicate]]$data$value -
              replicate_results[[replicate]]$data$value
          }
        ))
        replicate_diagnostics <- lapply(
          replicate_results, function(result) result$diagnostics
        )
      }

      effect_summary <- .mc_effect_summary_mst_pmdn(effect_values)
      output_index <- output_index + 1L
      block <- data.frame(
        case = cases,
        group = group_name,
        patch = patch_table$patch[patch_row],
        row_start = patch_table$row_start[patch_row],
        row_end = patch_table$row_end[patch_row],
        col_start = patch_table$col_start[patch_row],
        col_end = patch_table$col_end[patch_row],
        row_center = patch_table$row_center[patch_row],
        col_center = patch_table$col_center[patch_row],
        effect = effect_summary$effect,
        mc_effect_sd = effect_summary$mc_effect_sd,
        mc_effect_min = effect_summary$mc_effect_min,
        mc_effect_max = effect_summary$mc_effect_max,
        mc_sign_stable = effect_summary$mc_sign_stable,
        mc_replicates = replicate_count
      )
      if (isTRUE(decompose)) {
        for (channel in requested_channels) {
          block[[paste0("channel_", channel)]] <-
            rowMeans(channel_matrices[[channel]])
        }
        block$sum_to_total_residual <- block$effect - rowSums(
          block[paste0("channel_", requested_channels)]
        )
      } else {
        block$sum_to_total_residual <- NA_real_
      }
      rows_out[[output_index]] <- block

      replicate_block <- do.call(rbind, lapply(
        seq_len(replicate_count),
        function(replicate) {
          item <- data.frame(
            case = cases,
            group = group_name,
            patch = patch_table$patch[patch_row],
            row_center = patch_table$row_center[patch_row],
            col_center = patch_table$col_center[patch_row],
            replicate = replicate,
            effect = effect_values[, replicate]
          )
          if (isTRUE(decompose)) {
            for (channel in requested_channels) {
              item[[paste0("channel_", channel)]] <-
                channel_matrices[[channel]][, replicate]
            }
            item$sum_to_total_residual <- item$effect - rowSums(
              item[paste0("channel_", requested_channels)]
            )
          } else {
            item$sum_to_total_residual <- NA_real_
          }
          item
        }
      ))
      rownames(replicate_block) <- NULL
      replicate_rows_out[[output_index]] <- replicate_block
      diagnostics[[output_index]] <- c(
        list(replicate = replicate_diagnostics),
        .tail_resolution_summary_mst_pmdn(
          replicate_diagnostics, min_tail_draws
        )
      )
    }
  }

  data <- do.call(rbind, rows_out)
  replicate_data <- do.call(rbind, replicate_rows_out)
  rownames(data) <- NULL
  rownames(replicate_data) <- NULL
  inactive_channels <- setdiff(requested_channels, active_channels)
  if (length(inactive_channels)) {
    data[paste0("channel_", inactive_channels)] <- NULL
    replicate_data[paste0("channel_", inactive_channels)] <- NULL
  }
  if (length(active_channels)) {
    data$sum_to_total_residual <- data$effect - rowSums(
      data[paste0("channel_", active_channels)]
    )
    replicate_data$sum_to_total_residual <-
      replicate_data$effect - rowSums(
        replicate_data[paste0("channel_", active_channels)]
      )
  }

  population_replicate <- do.call(rbind, lapply(
    split(
      replicate_data,
      list(
        replicate_data$group,
        replicate_data$patch,
        replicate_data$replicate
      ),
      drop = TRUE
    ),
    function(block) data.frame(
      group = block$group[1L],
      patch = block$patch[1L],
      row_center = block$row_center[1L],
      col_center = block$col_center[1L],
      replicate = block$replicate[1L],
      mean_signed_effect = mean(block$effect)
    )
  ))
  rownames(population_replicate) <- NULL
  population <- do.call(rbind, lapply(
    split(data, list(data$group, data$patch), drop = TRUE),
    function(block) {
      replicate_block <- population_replicate[
        population_replicate$group == block$group[1L] &
          population_replicate$patch == block$patch[1L],
        ,
        drop = FALSE
      ]
      mc_summary <- .mc_effect_summary_mst_pmdn(matrix(
        replicate_block$mean_signed_effect,
        nrow = 1L
      ))
      data.frame(
        group = block$group[1L],
        patch = block$patch[1L],
        row_center = block$row_center[1L],
        col_center = block$col_center[1L],
        mean_signed_effect = mean(block$effect),
        mean_absolute_effect = mean(abs(block$effect)),
        positive_fraction = mean(block$effect > 0),
        mc_mean_signed_effect_sd = mc_summary$mc_effect_sd,
        mc_mean_signed_effect_min = mc_summary$mc_effect_min,
        mc_mean_signed_effect_max = mc_summary$mc_effect_max,
        mc_mean_signed_effect_sign_stable = mc_summary$mc_sign_stable,
        mc_replicates = replicate_count
      )
    }
  ))
  rownames(population) <- NULL

  original_diagnostics <- lapply(
    original_results, function(result) result$diagnostics
  )
  tail_summary <- .tail_resolution_summary_mst_pmdn(
    c(original_diagnostics, diagnostics),
    min_tail_draws
  )
  .warn_tail_resolution_mst_pmdn(
    tail_summary, "image_occlusion_mst_pmdn()"
  )
  minimum_tail_by_evaluation <- c(
    vapply(
      original_diagnostics,
      function(item) item$min_expected_tail_draws,
      numeric(1)
    ),
    unlist(lapply(diagnostics, function(item) {
      vapply(
        item$replicate,
        function(replicate) replicate$min_expected_tail_draws,
        numeric(1)
      )
    }), use.names = FALSE)
  )

  out <- list(
    data = data,
    replicate_data = replicate_data,
    population = population,
    population_replicate = population_replicate,
    patches = patch_table,
    coverage = coverage,
    weighted_coverage = weighted_coverage,
    functional = functional,
    cases = cases,
    channel_groups = groups,
    active_channels = active_channels,
    settings = list(
      patch_size = patch_size,
      stride = stride,
      taper = taper,
      decomposed = isTRUE(decompose),
      rebuild_channels = !is.null(rebuild_channels),
      num_samples = if (is.null(latent_banks[[1L]])) {
        NA_integer_
      } else {
        latent_banks[[1L]]$num_samples
      },
      mc_replicates = replicate_count,
      chunk_size = chunk_size,
      device = device,
      min_tail_draws = min_tail_draws
    ),
    diagnostics = c(
      list(
        original = if (replicate_count == 1L) {
          original_diagnostics[[1L]]
        } else {
          original_diagnostics
        },
        occluded = diagnostics,
        min_expected_tail_draws_by_evaluation =
          minimum_tail_by_evaluation,
        max_abs_sum_to_total_residual = if (length(active_channels)) {
          .max_abs_finite_mst_pmdn(data$sum_to_total_residual)
        } else {
          NA_real_
        }
      ),
      tail_summary
    ),
    latent_draws = .latent_banks_for_output_mst_pmdn(latent_banks)
  )
  class(out) <- "mst_pmdn_image_occlusion"
  out
}

as.data.frame.mst_pmdn_image_contrast <- function(x, ...) x$data
as.data.frame.mst_pmdn_image_occlusion <- function(x, ...) x$data

print.mst_pmdn_image_contrast <- function(x, ...) {
  cat(
    "MST-PMDN whole-image functional contrast:\n",
    "  cases: ", nrow(x$data), "\n",
    "  decomposed: ", x$settings$decomposed, "\n",
    sep = ""
  )
  invisible(x)
}

print.mst_pmdn_image_occlusion <- function(x, ...) {
  cat(
    "MST-PMDN image occlusion effects:\n",
    "  cases: ", length(x$cases), "\n",
    "  patches: ", nrow(x$patches), "\n",
    "  channel groups: ", length(x$channel_groups), "\n",
    "  Monte Carlo replicates: ", x$settings$mc_replicates, "\n",
    "  decomposed: ", x$settings$decomposed, "\n",
    sep = ""
  )
  invisible(x)
}
