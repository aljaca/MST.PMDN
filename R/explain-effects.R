################################################################################
# Tabular accumulated local effects and centred ICE                            #
################################################################################

.as_input_matrix_mst_pmdn <- function(inputs) {
  if (inherits(inputs, "torch_tensor")) {
    size <- as.integer(inputs$size())
    if (length(size) != 2L) {
      stop("inputs must be a two-dimensional tensor, matrix, or data frame.",
           call. = FALSE)
    }
    out <- as.matrix(torch::as_array(inputs$to(device = "cpu")))
  } else if (is.data.frame(inputs) || is.matrix(inputs)) {
    if (is.data.frame(inputs) && !all(vapply(inputs, is.numeric, logical(1)))) {
      stop("Every column of inputs must be numeric.", call. = FALSE)
    }
    out <- as.matrix(inputs)
  } else {
    stop("inputs must be a two-dimensional tensor, matrix, or data frame.",
         call. = FALSE)
  }
  if (!is.numeric(out) || nrow(out) < 1L || ncol(out) < 1L) {
    stop("inputs must contain numeric rows and columns.", call. = FALSE)
  }
  out
}

.resolve_feature_mst_pmdn <- function(feature, inputs_matrix) {
  if (is.character(feature)) {
    if (length(feature) != 1L || is.na(feature) ||
        is.null(colnames(inputs_matrix))) {
      stop("A named feature requires named input columns.", call. = FALSE)
    }
    index <- match(feature, colnames(inputs_matrix))
    if (is.na(index)) {
      stop(sprintf("Feature '%s' is not an input column.", feature),
           call. = FALSE)
    }
    return(list(index = as.integer(index), name = feature))
  }
  if (!is.numeric(feature) || length(feature) != 1L ||
      !is.finite(feature) || feature < 1 || feature != floor(feature) ||
      feature > ncol(inputs_matrix)) {
    stop("feature must be a valid 1-based column index or column name.",
         call. = FALSE)
  }
  index <- as.integer(feature)
  name <- if (is.null(colnames(inputs_matrix))) {
    paste0("feature_", index)
  } else {
    colnames(inputs_matrix)[index]
  }
  list(index = index, name = name)
}

.subset_rows_mst_pmdn <- function(x, rows) {
  if (is.null(x)) return(NULL)
  rows <- as.integer(rows)
  if (inherits(x, "torch_tensor")) {
    index <- torch_tensor(rows, dtype = torch_long(), device = x$device)
    return(torch_index_select(x, 1L, index))
  }
  dimensions <- dim(x)
  if (is.null(dimensions) || length(dimensions) < 2L) {
    stop("Case-matched inputs must have a leading row dimension.",
         call. = FALSE)
  }
  arguments <- c(
    list(x, rows),
    rep(list(TRUE), length(dimensions) - 1L),
    list(drop = FALSE)
  )
  do.call(`[`, arguments)
}

.n_rows_mst_pmdn <- function(x) {
  if (inherits(x, "torch_tensor")) return(as.integer(x$size(1L)))
  dim(x)[1L]
}

.validate_image_alignment_mst_pmdn <- function(image_inputs, n) {
  if (is.null(image_inputs)) return(invisible(TRUE))
  if (.n_rows_mst_pmdn(image_inputs) != n) {
    stop("image_inputs must have one case-matched row per tabular input row.",
         call. = FALSE)
  }
  invisible(TRUE)
}

.predict_chunks_mst_pmdn <- function(model,
                                     inputs,
                                     image_inputs = NULL,
                                     chunk_size = NULL,
                                     device = "cpu") {
  n <- nrow(inputs)
  .validate_image_alignment_mst_pmdn(image_inputs, n)
  if (is.null(chunk_size)) chunk_size <- n
  chunk_size <- validate_num_samples(chunk_size)
  starts <- seq.int(1L, n, by = chunk_size)
  predictions <- lapply(starts, function(start) {
    rows <- seq.int(start, min(n, start + chunk_size - 1L))
    predict_mst_pmdn(
      model,
      new_inputs = inputs[rows, , drop = FALSE],
      image_inputs = .subset_rows_mst_pmdn(image_inputs, rows),
      device = device
    )
  })
  if (length(predictions) == 1L) predictions[[1L]] else
    .bind_predictions_mst_pmdn(predictions)
}

.ensure_latent_bank_mst_pmdn <- function(pred,
                                         functional,
                                         latent_draws,
                                         num_samples,
                                         seed,
                                         device) {
  if (!.functional_is_monte_carlo_mst_pmdn(functional$type)) {
    return(NULL)
  }
  info <- .validate_prediction_mst_pmdn(pred)
  if (is.null(latent_draws)) {
    return(latent_draws_mst_pmdn(
      num_samples = num_samples,
      output_dim = info$output_dim,
      dtype = pred$mu$dtype,
      device = device,
      seed = seed
    ))
  }
  .validate_latent_draws_mst_pmdn(
    latent_draws,
    output_dim = info$output_dim,
    dtype = pred$mu$dtype,
    device = device
  )
}

.functional_values_quiet_mst_pmdn <- function(pred,
                                               functional,
                                               num_samples,
                                               latent_draws,
                                               chunk_size,
                                               device,
                                               response_names) {
  suppressWarnings(functional_mst_pmdn(
    pred = pred,
    functional = functional,
    num_samples = num_samples,
    latent_draws = latent_draws,
    chunk_size = chunk_size,
    device = device,
    response_names = response_names
  ))
}

.empirical_breaks_mst_pmdn <- function(x, n_bins) {
  if (any(!is.finite(x))) {
    stop("The interpreted feature must contain only finite values.",
         call. = FALSE)
  }
  n_bins <- min(validate_num_samples(n_bins), length(x))
  breaks <- unique(as.numeric(stats::quantile(
    x,
    probs = seq(0, 1, length.out = n_bins + 1L),
    names = FALSE,
    type = 7
  )))
  if (length(breaks) < 2L) {
    stop("The interpreted feature must have at least two distinct values.",
         call. = FALSE)
  }
  breaks
}

.bin_index_mst_pmdn <- function(x, breaks) {
  bins <- findInterval(x, breaks, all.inside = TRUE)
  pmin(bins, length(breaks) - 1L)
}

.nonempty_ale_bins_mst_pmdn <- function(x, breaks) {
  repeat {
    bins <- .bin_index_mst_pmdn(x, breaks)
    counts <- tabulate(bins, nbins = length(breaks) - 1L)
    empty <- which(counts == 0L)
    if (!length(empty)) {
      return(list(breaks = breaks, bins = bins, counts = counts))
    }
    k <- empty[1L]
    remove <- if (k < length(breaks) - 1L) k + 1L else k
    breaks <- breaks[-remove]
    if (length(breaks) < 2L) {
      stop("Unable to construct non-empty empirical ALE bins.",
           call. = FALSE)
    }
  }
}

# One-dimensional accumulated local effects for MST-PMDN functionals
ale_mst_pmdn <- function(model,
                         inputs,
                         feature,
                         functional,
                         image_inputs = NULL,
                         n_bins = 20L,
                         num_samples = 4096L,
                         decompose = FALSE,
                         channels = c("location", "scale", "skewness", "df"),
                         latent_draws = NULL,
                         seed = NULL,
                         chunk_size = NULL,
                         device = "cpu",
                         response_names = NULL) {
  if (!inherits(functional, "mst_functional")) {
    stop("functional must be returned by mst_functional().", call. = FALSE)
  }
  inputs_matrix <- .as_input_matrix_mst_pmdn(inputs)
  feature_info <- .resolve_feature_mst_pmdn(feature, inputs_matrix)
  .validate_image_alignment_mst_pmdn(image_inputs, nrow(inputs_matrix))
  num_samples <- validate_num_samples(num_samples)
  if (isTRUE(decompose)) {
    probe <- .predict_chunks_mst_pmdn(
      model,
      inputs_matrix[1L, , drop = FALSE],
      .subset_rows_mst_pmdn(image_inputs, 1L),
      chunk_size = 1L,
      device = device
    )
    .require_single_component_mst_pmdn(probe)
  }
  breaks <- .empirical_breaks_mst_pmdn(
    inputs_matrix[, feature_info$index], n_bins
  )
  binning <- .nonempty_ale_bins_mst_pmdn(
    inputs_matrix[, feature_info$index], breaks
  )
  breaks <- binning$breaks
  bins <- binning$bins
  K <- length(breaks) - 1L
  counts <- binning$counts
  bin_effect <- numeric(K)
  bin_channel_effects <- vector("list", K)
  active_channels <- character(0)
  functional_diagnostics <- vector("list", K)

  for (k in seq_len(K)) {
    rows <- which(bins == k)
    x_low <- inputs_matrix[rows, , drop = FALSE]
    x_high <- x_low
    x_low[, feature_info$index] <- breaks[k]
    x_high[, feature_info$index] <- breaks[k + 1L]
    image_rows <- .subset_rows_mst_pmdn(image_inputs, rows)
    pred_low <- .predict_chunks_mst_pmdn(
      model, x_low, image_rows, chunk_size = chunk_size, device = device
    )
    pred_high <- .predict_chunks_mst_pmdn(
      model, x_high, image_rows, chunk_size = chunk_size, device = device
    )
    latent_draws <- .ensure_latent_bank_mst_pmdn(
      pred_low, functional, latent_draws, num_samples, seed, device
    )

    if (isTRUE(decompose)) {
      decomposition <- suppressWarnings(decompose_mst_pmdn(
        pred_from = pred_low,
        pred_to = pred_high,
        functional = functional,
        channels = channels,
        method = "shapley",
        latent_draws = latent_draws,
        num_samples = num_samples,
        chunk_size = chunk_size,
        device = device,
        response_names = response_names
      ))
      local_effect <- decomposition$data$total
      active_channels <- union(
        active_channels, decomposition$active_channels
      )
      bin_channel_effects[[k]] <- lapply(
        decomposition$active_channels,
        function(channel) decomposition$data[[paste0("channel_", channel)]]
      )
      names(bin_channel_effects[[k]]) <- decomposition$active_channels
      functional_diagnostics[[k]] <- list(
        min_expected_tail_draws =
          decomposition$diagnostics$min_expected_tail_draws,
        decomposition = decomposition$diagnostics
      )
    } else {
      low <- .functional_values_quiet_mst_pmdn(
        pred_low, functional, num_samples, latent_draws,
        chunk_size, device, response_names
      )
      high <- .functional_values_quiet_mst_pmdn(
        pred_high, functional, num_samples, latent_draws,
        chunk_size, device, response_names
      )
      local_effect <- high$data$value - low$data$value
      functional_diagnostics[[k]] <- list(
        min_expected_tail_draws = .min_finite_mst_pmdn(c(
          low$data$expected_tail_draws,
          high$data$expected_tail_draws
        )),
        low = low$diagnostics,
        high = high$diagnostics
      )
    }
    bin_effect[k] <- mean(local_effect)
  }

  midpoint_effect <- cumsum(bin_effect) - 0.5 * bin_effect
  centred_effect <- midpoint_effect - stats::weighted.mean(
    midpoint_effect, counts
  )
  data <- data.frame(
    bin = seq_len(K),
    lower = breaks[-length(breaks)],
    upper = breaks[-1L],
    feature_value = (breaks[-length(breaks)] + breaks[-1L]) / 2,
    ale = centred_effect,
    n = counts
  )

  if (isTRUE(decompose) && length(active_channels)) {
    for (channel in active_channels) {
      bin_mean <- vapply(seq_len(K), function(k) {
        values <- bin_channel_effects[[k]][[channel]]
        if (is.null(values)) 0 else mean(values)
      }, numeric(1))
      midpoint <- cumsum(bin_mean) - 0.5 * bin_mean
      data[[paste0("ale_", channel)]] <- midpoint -
        stats::weighted.mean(midpoint, counts)
    }
    channel_columns <- paste0("ale_", active_channels)
    data$sum_to_total_residual <- data$ale - rowSums(data[channel_columns])
  } else {
    data$sum_to_total_residual <- NA_real_
  }

  is_mc <- .functional_is_monte_carlo_mst_pmdn(functional$type)
  actual_samples <- if (!is.null(latent_draws)) {
    latent_draws$num_samples
  } else {
    NA_integer_
  }
  minimum_tail_by_bin <- if (is_mc) {
    vapply(
      functional_diagnostics,
      function(x) x$min_expected_tail_draws,
      numeric(1)
    )
  } else {
    rep(NA_real_, K)
  }
  out <- list(
    data = data,
    breaks = breaks,
    feature = feature_info,
    functional = functional,
    active_channels = active_channels,
    settings = list(
      requested_bins = as.integer(n_bins),
      actual_bins = K,
      num_samples = actual_samples,
      chunk_size = chunk_size,
      device = device,
      decomposed = isTRUE(decompose)
    ),
    diagnostics = list(
      bin = functional_diagnostics,
      min_expected_tail_draws_by_bin = minimum_tail_by_bin,
      min_expected_tail_draws = .min_finite_mst_pmdn(
        minimum_tail_by_bin
      ),
      max_abs_sum_to_total_residual = if (length(active_channels)) {
        .max_abs_finite_mst_pmdn(data$sum_to_total_residual)
      } else {
        NA_real_
      }
    ),
    latent_draws = latent_draws
  )
  class(out) <- "mst_pmdn_ale"
  out
}

.ice_slopes_mst_pmdn <- function(grid, values) {
  n <- length(grid)
  if (n < 2L) return(rep(NA_real_, n))
  slopes <- numeric(n)
  slopes[1L] <- (values[2L] - values[1L]) / (grid[2L] - grid[1L])
  slopes[n] <- (values[n] - values[n - 1L]) / (grid[n] - grid[n - 1L])
  if (n > 2L) {
    index <- 2:(n - 1L)
    slopes[index] <- (values[index + 1L] - values[index - 1L]) /
      (grid[index + 1L] - grid[index - 1L])
  }
  slopes
}

# Centred individual conditional expectation for MST-PMDN functionals
ice_mst_pmdn <- function(model,
                         inputs,
                         feature,
                         functional,
                         image_inputs = NULL,
                         grid = NULL,
                         reference = NULL,
                         n_curves = 100L,
                         derivative = FALSE,
                         ale = TRUE,
                         n_bins = 20L,
                         num_samples = 4096L,
                         latent_draws = NULL,
                         seed = NULL,
                         chunk_size = NULL,
                         device = "cpu",
                         response_names = NULL) {
  if (!inherits(functional, "mst_functional")) {
    stop("functional must be returned by mst_functional().", call. = FALSE)
  }
  inputs_matrix <- .as_input_matrix_mst_pmdn(inputs)
  feature_info <- .resolve_feature_mst_pmdn(feature, inputs_matrix)
  .validate_image_alignment_mst_pmdn(image_inputs, nrow(inputs_matrix))
  n_curves <- min(validate_num_samples(n_curves), nrow(inputs_matrix))
  case_rows <- unique(as.integer(round(seq(
    1, nrow(inputs_matrix), length.out = n_curves
  ))))
  feature_values <- inputs_matrix[, feature_info$index]
  if (is.null(grid)) {
    grid <- unique(as.numeric(stats::quantile(
      feature_values,
      probs = seq(0.05, 0.95, length.out = 25L),
      names = FALSE,
      type = 7
    )))
  }
  if (!is.numeric(grid) || length(grid) < 2L || any(!is.finite(grid))) {
    stop("grid must contain at least two finite numeric values.",
         call. = FALSE)
  }
  grid <- sort(unique(as.numeric(grid)))
  if (length(grid) < 2L) {
    stop("grid must contain at least two distinct values.", call. = FALSE)
  }
  if (is.null(reference)) reference <- stats::median(feature_values)
  if (!is.numeric(reference) || length(reference) != 1L ||
      !is.finite(reference)) {
    stop("reference must be one finite feature value.", call. = FALSE)
  }
  num_samples <- validate_num_samples(num_samples)

  selected_inputs <- inputs_matrix[case_rows, , drop = FALSE]
  selected_images <- .subset_rows_mst_pmdn(image_inputs, case_rows)
  reference_inputs <- selected_inputs
  reference_inputs[, feature_info$index] <- reference
  pred_reference <- .predict_chunks_mst_pmdn(
    model, reference_inputs, selected_images,
    chunk_size = chunk_size, device = device
  )
  latent_draws <- .ensure_latent_bank_mst_pmdn(
    pred_reference, functional, latent_draws, num_samples, seed, device
  )
  reference_result <- .functional_values_quiet_mst_pmdn(
    pred_reference, functional, num_samples, latent_draws,
    chunk_size, device, response_names
  )
  reference_values <- reference_result$data$value
  pred_original <- .predict_chunks_mst_pmdn(
    model, selected_inputs, selected_images,
    chunk_size = chunk_size, device = device
  )
  original_result <- .functional_values_quiet_mst_pmdn(
    pred_original, functional, num_samples, latent_draws,
    chunk_size, device, response_names
  )

  values <- matrix(NA_real_, nrow = length(case_rows), ncol = length(grid))
  grid_diagnostics <- vector("list", length(grid))
  for (g in seq_along(grid)) {
    grid_inputs <- selected_inputs
    grid_inputs[, feature_info$index] <- grid[g]
    pred_grid <- .predict_chunks_mst_pmdn(
      model, grid_inputs, selected_images,
      chunk_size = chunk_size, device = device
    )
    result <- .functional_values_quiet_mst_pmdn(
      pred_grid, functional, num_samples, latent_draws,
      chunk_size, device, response_names
    )
    values[, g] <- result$data$value
    grid_diagnostics[[g]] <- result$diagnostics
  }
  centred <- sweep(values, 1L, reference_values, "-")
  slopes <- matrix(NA_real_, nrow = nrow(centred), ncol = ncol(centred))
  if (isTRUE(derivative)) {
    for (i in seq_len(nrow(centred))) {
      slopes[i, ] <- .ice_slopes_mst_pmdn(grid, centred[i, ])
    }
  }

  curves <- do.call(rbind, lapply(seq_along(case_rows), function(i) {
    data.frame(
      case = case_rows[i],
      feature_value = grid,
      value = values[i, ],
      centred = centred[i, ],
      slope = slopes[i, ]
    )
  }))
  plate <- do.call(rbind, lapply(seq_along(case_rows), function(i) {
    x_i <- inputs_matrix[case_rows[i], feature_info$index]
    data.frame(
      case = case_rows[i],
      feature_value = x_i,
      baseline_contrast = original_result$data$value[i] - reference_values[i],
      local_slope = if (isTRUE(derivative)) {
        stats::approx(grid, slopes[i, ], xout = x_i, rule = 2)$y
      } else {
        NA_real_
      }
    )
  }))
  if (inherits(ale, "mst_pmdn_ale")) {
    if (!identical(ale$feature$index, feature_info$index) ||
        !identical(ale$functional, functional)) {
      stop("A supplied ALE object must match feature and functional.",
           call. = FALSE)
    }
    ale_result <- ale
    ale_mode <- "supplied"
  } else if (isTRUE(ale)) {
    ale_result <- ale_mst_pmdn(
      model = model,
      inputs = inputs_matrix,
      image_inputs = image_inputs,
      feature = feature_info$index,
      functional = functional,
      n_bins = n_bins,
      num_samples = num_samples,
      decompose = FALSE,
      latent_draws = latent_draws,
      chunk_size = chunk_size,
      device = device,
      response_names = response_names
    )
    ale_mode <- "computed"
  } else if (identical(ale, FALSE) || is.null(ale)) {
    ale_result <- NULL
    ale_mode <- "none"
  } else {
    stop("ale must be TRUE, FALSE, NULL, or an mst_pmdn_ale object.",
         call. = FALSE)
  }

  out <- list(
    curves = curves,
    plate = plate,
    ale = ale_result,
    feature = feature_info,
    functional = functional,
    grid = grid,
    reference = reference,
    cases = case_rows,
    settings = list(
      derivative = isTRUE(derivative),
      ale = ale_mode,
      num_samples = if (is.null(latent_draws)) NA_integer_ else
        latent_draws$num_samples,
      chunk_size = chunk_size,
      device = device
    ),
    diagnostics = list(
      reference = reference_result$diagnostics,
      original = original_result$diagnostics,
      grid = grid_diagnostics
    ),
    latent_draws = latent_draws
  )
  class(out) <- "mst_pmdn_ice"
  out
}

as.data.frame.mst_pmdn_ale <- function(x, ...) x$data
as.data.frame.mst_pmdn_ice <- function(x, ...) x$curves

print.mst_pmdn_ale <- function(x, ...) {
  cat(
    "MST-PMDN accumulated local effects:\n",
    "  feature: ", x$feature$name, " (", x$feature$index, ")\n",
    "  bins: ", nrow(x$data), "\n",
    "  decomposed: ", x$settings$decomposed, "\n",
    sep = ""
  )
  invisible(x)
}

print.mst_pmdn_ice <- function(x, ...) {
  cat(
    "MST-PMDN centred ICE:\n",
    "  feature: ", x$feature$name, " (", x$feature$index, ")\n",
    "  curves: ", length(x$cases), "\n",
    "  grid points: ", length(x$grid), "\n",
    sep = ""
  )
  invisible(x)
}
