################################################################################
# Single-component distributional parameter-channel decomposition              #
################################################################################

.validate_channels_mst_pmdn <- function(channels) {
  allowed <- c("location", "scale", "skewness", "df")
  if (!is.character(channels) || !length(channels) || anyNA(channels) ||
      any(!channels %in% allowed)) {
    stop(
      "channels must select location, scale, skewness, and/or df.",
      call. = FALSE
    )
  }
  unique(channels)
}

.tensor_equal_mst_pmdn <- function(x, y) {
  if (!inherits(x, "torch_tensor") || !inherits(y, "torch_tensor")) {
    return(identical(x, y))
  }
  if (!identical(as.integer(x$size()), as.integer(y$size())) ||
      x$dtype != y$dtype) {
    return(FALSE)
  }
  isTRUE(torch_equal(x$to(device = "cpu"), y$to(device = "cpu")))
}

.effective_skew_none_mst_pmdn <- function(pred, name) {
  .validate_skew_none(pred, name) ||
    isTRUE((pred$alpha == 0)$all()$item())
}

.tensor_change_magnitude_mst_pmdn <- function(from, to) {
  from_value <- as.numeric(torch::as_array(from$to(device = "cpu")))
  to_value <- as.numeric(torch::as_array(to$to(device = "cpu")))
  .max_abs_finite_mst_pmdn(to_value - from_value)
}

.inverse_nu_mst_pmdn <- function(nu) {
  torch_where(nu == Inf, torch_zeros_like(nu), 1 / nu)
}

.channel_change_magnitudes_mst_pmdn <- function(pred_from, pred_to) {
  c(
    location = .tensor_change_magnitude_mst_pmdn(
      pred_from$mu, pred_to$mu
    ),
    scale = .tensor_change_magnitude_mst_pmdn(
      pred_from$scale_chol, pred_to$scale_chol
    ),
    skewness = .tensor_change_magnitude_mst_pmdn(
      pred_from$alpha, pred_to$alpha
    ),
    df = .tensor_change_magnitude_mst_pmdn(
      .inverse_nu_mst_pmdn(pred_from$nu),
      .inverse_nu_mst_pmdn(pred_to$nu)
    )
  )
}

.channel_is_active_mst_pmdn <- function(channel, pred_from, pred_to) {
  switch(
    channel,
    location = !.tensor_equal_mst_pmdn(pred_from$mu, pred_to$mu),
    scale = !.tensor_equal_mst_pmdn(
      pred_from$scale_chol, pred_to$scale_chol
    ),
    skewness = {
      from_none <- .effective_skew_none_mst_pmdn(
        pred_from, "pred_from"
      )
      to_none <- .effective_skew_none_mst_pmdn(
        pred_to, "pred_to"
      )
      !(from_none && to_none) &&
        (!identical(from_none, to_none) ||
         !.tensor_equal_mst_pmdn(pred_from$alpha, pred_to$alpha))
    },
    df = !.tensor_equal_mst_pmdn(pred_from$nu, pred_to$nu),
    FALSE
  )
}

.hybrid_prediction_mst_pmdn <- function(pred_from,
                                        pred_to,
                                        selected_channels,
                                        gamma_cache_key = NULL) {
  out <- pred_from
  if ("location" %in% selected_channels) out$mu <- pred_to$mu
  if ("scale" %in% selected_channels) {
    out$scale_chol <- pred_to$scale_chol
  }
  if ("skewness" %in% selected_channels) {
    out$alpha <- pred_to$alpha
    # Assigning NULL deliberately removes the optional legacy marker; the
    # validator then infers the skew path from alpha.
    if (is.null(pred_to$skew_none)) {
      out$skew_none <- NULL
    } else {
      out$skew_none <- pred_to$skew_none
    }
  }
  if ("df" %in% selected_channels) out$nu <- pred_to$nu
  if (!is.null(gamma_cache_key)) {
    out$.gamma_cache_key <- gamma_cache_key
  }
  out
}

.channels_from_mask_mst_pmdn <- function(mask, channels) {
  if (!length(channels)) return(character(0))
  bits <- bitwShiftL(1L, seq_along(channels) - 1L)
  channels[bitwAnd(mask, bits) != 0L]
}

.known_functional_chunk_mst_pmdn <- function(result, rows) {
  out <- result
  out$data <- result$data[rows, , drop = FALSE]
  out$data$row <- seq_along(rows)
  out$diagnostics$low_tail_resolution_rows <- which(
    out$data$low_tail_resolution
  )
  out$diagnostics$min_expected_tail_draws <- .min_finite_mst_pmdn(
    out$data$expected_tail_draws
  )
  out
}

.combine_state_chunks_mst_pmdn <- function(chunks) {
  diagnostics <- chunks[[1L]]$result$diagnostics
  diagnostics$low_tail_resolution_rows <- as.integer(unlist(lapply(
    chunks,
    function(chunk) {
      chunk$rows[chunk$result$data$low_tail_resolution]
    }
  )))
  diagnostics$min_expected_tail_draws <- .min_finite_mst_pmdn(vapply(
    chunks,
    function(chunk) chunk$result$diagnostics$min_expected_tail_draws,
    numeric(1)
  ))
  diagnostics$chunk <- lapply(chunks, function(chunk) {
    chunk$result$diagnostics
  })
  diagnostics
}

.require_single_component_mst_pmdn <- function(pred) {
  info <- .validate_prediction_mst_pmdn(pred)
  if (info$n_mixtures != 1L) {
    stop(
      paste0(
        "Full parameter-channel decomposition is only available for M = 1. ",
        "Use functional effects or tail_components_mst_pmdn() for mixtures."
      ),
      call. = FALSE
    )
  }
  invisible(info)
}

# Decompose a one-component functional contrast among parameter channels
.decompose_mst_pmdn_impl <- function(pred_from,
                               pred_to,
                               functional,
                               channels = c(
                                 "location", "scale", "skewness", "df"
                               ),
                               method = "shapley",
                               latent_draws = NULL,
                               num_samples = 4096L,
                               seed = NULL,
                               chunk_size = NULL,
                               device = "cpu",
                               response_names = NULL,
                               min_tail_draws = 20L,
                               .known_to_result = NULL) {
  from_info <- .validate_prediction_mst_pmdn(pred_from, "pred_from")
  to_info <- .validate_prediction_mst_pmdn(pred_to, "pred_to")
  if (!identical(from_info, to_info)) {
    stop("pred_from and pred_to must have matching prediction dimensions.",
         call. = FALSE)
  }
  .require_single_component_mst_pmdn(pred_from)
  if (!inherits(functional, "mst_functional")) {
    stop("functional must be returned by mst_functional().", call. = FALSE)
  }
  if (!is.null(.known_to_result) &&
      (!inherits(.known_to_result, "mst_pmdn_functional") ||
       nrow(.known_to_result$data) != from_info$batch_size ||
       !identical(.known_to_result$functional, functional))) {
    stop("Internal known endpoint does not match the decomposition.",
         call. = FALSE)
  }
  method <- match.arg(method, "shapley")
  channels <- .validate_channels_mst_pmdn(channels)
  active_channels <- channels[vapply(
    channels,
    .channel_is_active_mst_pmdn,
    logical(1),
    pred_from = pred_from,
    pred_to = pred_to
  )]
  num_samples <- validate_num_samples(num_samples)
  min_tail_draws <- validate_num_samples(min_tail_draws)
  latent_draws <- .ensure_latent_bank_mst_pmdn(
    pred_from, functional, latent_draws, num_samples, seed, device
  )
  if (!is.null(latent_draws)) num_samples <- latent_draws$num_samples

  K <- length(active_channels)
  n_states <- 2^K
  is_mc <- .functional_is_monte_carlo_mst_pmdn(functional$type)
  evaluation_chunk_size <- if (is.null(chunk_size)) {
    if (is_mc) {
      .auto_functional_chunk_size_mst_pmdn(
        from_info$batch_size, num_samples, from_info$output_dim
      )
    } else {
      from_info$batch_size
    }
  } else {
    validate_num_samples(chunk_size)
  }
  starts <- seq.int(
    1L, from_info$batch_size, by = evaluation_chunk_size
  )
  state_values <- matrix(
    NA_real_, nrow = n_states, ncol = from_info$batch_size
  )
  state_chunks <- lapply(seq_len(n_states), function(x) {
    vector("list", length(starts))
  })
  gamma_cache_scope <- if (is.null(latent_draws)) {
    NULL
  } else {
    .next_gamma_cache_scope_mst_pmdn(latent_draws$.cache)
  }
  full_state <- n_states - 1L

  for (chunk_index in seq_along(starts)) {
    start <- starts[chunk_index]
    rows <- seq.int(
      start,
      min(from_info$batch_size, start + evaluation_chunk_size - 1L)
    )
    from_chunk <- .slice_prediction_mst_pmdn(pred_from, rows)
    to_chunk <- .slice_prediction_mst_pmdn(pred_to, rows)

    for (state in 0:full_state) {
      selected <- .channels_from_mask_mst_pmdn(
        state, active_channels
      )
      if (!is.null(.known_to_result) && state == full_state) {
        result <- .known_functional_chunk_mst_pmdn(
          .known_to_result, rows
        )
      } else {
        nu_source <- if ("df" %in% selected) "to" else "from"
        gamma_cache_key <- if (is.null(gamma_cache_scope)) {
          NULL
        } else {
          paste(
            gamma_cache_scope,
            nu_source,
            rows[1L],
            rows[length(rows)],
            sep = ":"
          )
        }
        hybrid <- .hybrid_prediction_mst_pmdn(
          from_chunk,
          to_chunk,
          selected,
          gamma_cache_key = gamma_cache_key
        )
        result <- .functional_values_quiet_mst_pmdn(
          hybrid,
          functional,
          num_samples,
          latent_draws,
          length(rows),
          device,
          response_names,
          min_tail_draws
        )
      }
      state_values[state + 1L, rows] <- result$data$value
      state_chunks[[state + 1L]][[chunk_index]] <- list(
        rows = rows,
        result = result
      )
    }
  }
  state_results <- lapply(state_chunks, .combine_state_chunks_mst_pmdn)
  state_low_resolution <- vapply(
    state_chunks,
    function(chunks) sum(vapply(
      chunks,
      function(chunk) sum(chunk$result$data$low_tail_resolution),
      integer(1)
    )),
    integer(1)
  )
  reused_to_low_resolution <- if (is.null(.known_to_result)) {
    0L
  } else {
    sum(.known_to_result$data$low_tail_resolution)
  }

  contributions <- matrix(
    0,
    nrow = from_info$batch_size,
    ncol = K,
    dimnames = list(NULL, active_channels)
  )
  if (K > 0L) {
    full_mask <- n_states - 1L
    for (i in seq_len(K)) {
      bit <- bitwShiftL(1L, i - 1L)
      for (state in 0:full_mask) {
        if (bitwAnd(state, bit) != 0L) next
        subset_size <- sum(bitwAnd(
          state, bitwShiftL(1L, seq_len(K) - 1L)
        ) != 0L)
        weight <- factorial(subset_size) *
          factorial(K - subset_size - 1L) / factorial(K)
        without <- state_values[state + 1L, ]
        with <- state_values[bitwOr(state, bit) + 1L, ]
        contributions[, i] <- contributions[, i] + weight * (with - without)
      }
    }
  }

  from_value <- state_values[1L, ]
  to_value <- state_values[n_states, ]
  total <- to_value - from_value
  residual <- total - rowSums(contributions)
  state_tail_resolution <- vapply(
    state_results,
    function(x) x$min_expected_tail_draws,
    numeric(1)
  )
  data <- data.frame(
    row = seq_len(from_info$batch_size),
    from = from_value,
    to = to_value,
    total = total
  )
  for (channel in active_channels) {
    data[[paste0("channel_", channel)]] <- contributions[, channel]
  }
  data$sum_to_total_residual <- residual

  out <- list(
    data = data,
    functional = functional,
    requested_channels = channels,
    active_channels = active_channels,
    method = method,
    settings = list(
      evaluations = n_states,
      num_samples = if (is.null(latent_draws)) NA_integer_ else
        latent_draws$num_samples,
      chunk_size = chunk_size,
      evaluation_chunk_size = evaluation_chunk_size,
      chunks = length(starts),
      device = device,
      min_tail_draws = min_tail_draws
    ),
    diagnostics = list(
      max_abs_sum_to_total_residual = .max_abs_finite_mst_pmdn(residual),
      min_expected_tail_draws = .min_finite_mst_pmdn(
        state_tail_resolution
      ),
      low_tail_resolution_evaluations = sum(state_low_resolution),
      reused_to_endpoint = !is.null(.known_to_result),
      reused_to_low_tail_resolution_evaluations =
        reused_to_low_resolution,
      max_abs_parameter_change = .channel_change_magnitudes_mst_pmdn(
        pred_from, pred_to
      ),
      state = state_results
    ),
    latent_draws = .latent_draws_for_output_mst_pmdn(latent_draws)
  )
  class(out) <- "mst_pmdn_decomposition"
  .warn_tail_resolution_mst_pmdn(
    out$diagnostics$min_expected_tail_draws,
    min_tail_draws,
    out$diagnostics$low_tail_resolution_evaluations,
    "Parameter-channel decomposition"
  )
  out
}

decompose_mst_pmdn <- function(pred_from,
                               pred_to,
                               functional,
                               channels = c(
                                 "location", "scale", "skewness", "df"
                               ),
                               method = "shapley",
                               latent_draws = NULL,
                               num_samples = 4096L,
                               seed = NULL,
                               chunk_size = NULL,
                               device = "cpu",
                               response_names = NULL,
                               min_tail_draws = 20L) {
  .decompose_mst_pmdn_impl(
    pred_from = pred_from,
    pred_to = pred_to,
    functional = functional,
    channels = channels,
    method = method,
    latent_draws = latent_draws,
    num_samples = num_samples,
    seed = seed,
    chunk_size = chunk_size,
    device = device,
    response_names = response_names,
    min_tail_draws = min_tail_draws,
    .known_to_result = NULL
  )
}

as.data.frame.mst_pmdn_decomposition <- function(x, ...) x$data

print.mst_pmdn_decomposition <- function(x, ...) {
  cat(
    "MST-PMDN parameter-channel decomposition:\n",
    "  active channels: ",
    if (length(x$active_channels)) {
      paste(x$active_channels, collapse = ", ")
    } else {
      "none"
    },
    "\n",
    "  rows: ", nrow(x$data), "\n",
    "  maximum closure residual: ",
    format(x$diagnostics$max_abs_sum_to_total_residual, digits = 4),
    "\n",
    sep = ""
  )
  invisible(x)
}
