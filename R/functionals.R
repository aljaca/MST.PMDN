################################################################################
# Scalar predictive functionals and exact MST-PMDN moments                     #
################################################################################

.functional_type_alias_mst_pmdn <- function(type) {
  aliases <- c(
    var = "variance",
    cov = "covariance",
    exceed = "exceedance",
    joint_exceedance_probability = "joint_exceedance",
    inverse_nu = "inverse_df",
    nu = "df"
  )
  type <- gsub("-", "_", tolower(type), fixed = TRUE)
  if (type %in% names(aliases)) aliases[[type]] else type
}

#' Define a scalar scientific functional of an MST-PMDN prediction
mst_functional <- function(type,
                           responses = NULL,
                           prob = NULL,
                           threshold = NULL,
                           direction = "upper",
                           transform = "identity") {
  if (!is.character(type) || length(type) != 1L || is.na(type)) {
    stop("type must be a single character value.", call. = FALSE)
  }
  type <- .functional_type_alias_mst_pmdn(type)
  allowed <- c(
    "mean", "variance", "sd", "covariance", "correlation",
    "quantile", "exceedance", "joint_exceedance", "tail_spread",
    "tail_asymmetry", "df", "inverse_df", "skew_direction"
  )
  if (!type %in% allowed) {
    stop(
      sprintf("Unsupported functional type '%s'.", type),
      call. = FALSE
    )
  }

  arity <- if (type %in% c("covariance", "correlation", "joint_exceedance")) {
    2L
  } else if (type %in% c("df", "inverse_df")) {
    0L
  } else {
    1L
  }
  if (is.null(responses)) responses <- integer(0)
  if (!(is.numeric(responses) || is.character(responses)) ||
      length(responses) != arity || anyNA(responses)) {
    stop(
      sprintf("responses must have length %d for type = '%s'.", arity, type),
      call. = FALSE
    )
  }
  if (is.numeric(responses) &&
      (any(!is.finite(responses)) || any(responses < 1) ||
       any(responses != floor(responses)))) {
    stop("Numeric responses must use positive 1-based indices.", call. = FALSE)
  }

  if (type == "quantile") {
    if (!is.numeric(prob) || length(prob) != 1L || !is.finite(prob) ||
        prob <= 0 || prob >= 1) {
      stop("prob must be one value strictly between zero and one.",
           call. = FALSE)
    }
  } else if (type %in% c("tail_spread", "tail_asymmetry")) {
    if (!is.numeric(prob) || length(prob) != 2L || any(!is.finite(prob)) ||
        prob[1] <= 0 || prob[2] >= 1 || prob[1] >= prob[2]) {
      stop("prob must contain increasing lower and upper probabilities in (0, 1).",
           call. = FALSE)
    }
  } else if (!is.null(prob)) {
    stop(sprintf("prob is not used by type = '%s'.", type), call. = FALSE)
  }

  if (type == "exceedance") {
    if (!is.numeric(threshold) || length(threshold) != 1L ||
        !is.finite(threshold)) {
      stop("threshold must be one finite value for an exceedance.",
           call. = FALSE)
    }
  } else if (type == "joint_exceedance") {
    if (!is.numeric(threshold) || length(threshold) != 2L ||
        any(!is.finite(threshold))) {
      stop("threshold must contain two finite values for a joint exceedance.",
           call. = FALSE)
    }
  } else if (!is.null(threshold)) {
    stop(sprintf("threshold is not used by type = '%s'.", type),
         call. = FALSE)
  }

  if (type %in% c("exceedance", "joint_exceedance")) {
    if (length(direction) == 1L) direction <- rep(direction, arity)
    if (!is.character(direction) || length(direction) != arity ||
        any(!direction %in% c("upper", "lower"))) {
      stop("direction must contain 'upper' or 'lower' for each response.",
           call. = FALSE)
    }
  } else {
    direction <- character(0)
  }

  transform <- match.arg(
    transform,
    c("identity", "log", "logit", "fisher_z")
  )
  if (transform == "logit" &&
      !type %in% c("exceedance", "joint_exceedance")) {
    stop("The logit transform is only available for probability functionals.",
         call. = FALSE)
  }
  if (transform == "fisher_z" && type != "correlation") {
    stop("The fisher_z transform is only available for correlation.",
         call. = FALSE)
  }

  out <- list(
    type = type,
    responses = responses,
    prob = prob,
    threshold = threshold,
    direction = direction,
    transform = transform,
    scale = switch(
      transform,
      logit = "log-odds",
      fisher_z = "Fisher-z",
      "functional"
    )
  )
  class(out) <- "mst_functional"
  out
}

print.mst_functional <- function(x, ...) {
  response_text <- if (length(x$responses)) {
    paste(x$responses, collapse = ", ")
  } else {
    "none"
  }
  cat(
    "MST-PMDN functional:\n",
    "  type: ", x$type, "\n",
    "  responses: ", response_text, "\n",
    "  transform: ", x$transform, "\n",
    sep = ""
  )
  invisible(x)
}

.validate_prediction_mst_pmdn <- function(pred, name = "pred") {
  if (!is.list(pred)) {
    stop(sprintf("%s must be a prediction list.", name), call. = FALSE)
  }
  required <- c("pi", "mu", "scale_chol", "nu", "alpha")
  if (!all(required %in% names(pred)) ||
      !all(vapply(pred[required], inherits, logical(1),
                  what = "torch_tensor"))) {
    stop(
      sprintf("%s must contain pi, mu, scale_chol, nu, and alpha tensors.", name),
      call. = FALSE
    )
  }
  pi_size <- as.integer(pred$pi$size())
  mu_size <- as.integer(pred$mu$size())
  chol_size <- as.integer(pred$scale_chol$size())
  nu_size <- as.integer(pred$nu$size())
  alpha_size <- as.integer(pred$alpha$size())
  valid <- length(pi_size) == 2L && length(mu_size) == 3L &&
    length(chol_size) == 4L && length(nu_size) == 2L &&
    length(alpha_size) == 3L &&
    identical(mu_size[1:2], pi_size) &&
    identical(nu_size, pi_size) &&
    identical(alpha_size, mu_size) &&
    identical(chol_size[1:2], pi_size) &&
    chol_size[3] == mu_size[3] && chol_size[4] == mu_size[3]
  if (!valid) {
    stop(sprintf("%s contains inconsistent prediction tensor shapes.", name),
         call. = FALSE)
  }
  if (any(pi_size < 1L) || mu_size[3] < 1L) {
    stop(sprintf("%s must contain at least one row, component, and response.", name),
         call. = FALSE)
  }
  .require_alpha(pred, name)
  .validate_skew_none(pred, name)
  list(batch_size = pi_size[1], n_mixtures = pi_size[2], output_dim = mu_size[3])
}

.resolve_responses_mst_pmdn <- function(functional,
                                        output_dim,
                                        response_names = NULL) {
  responses <- functional$responses
  if (is.character(responses)) {
    if (is.null(response_names)) {
      stop(
        "Character responses require response_names or a response_names attribute on pred.",
        call. = FALSE
      )
    }
    if (length(response_names) != output_dim || anyDuplicated(response_names)) {
      stop("response_names must uniquely name every output dimension.",
           call. = FALSE)
    }
    resolved <- match(responses, response_names)
    if (anyNA(resolved)) {
      stop("A requested response name is not present in response_names.",
           call. = FALSE)
    }
    return(as.integer(resolved))
  }
  responses <- as.integer(responses)
  if (length(responses) && any(responses > output_dim)) {
    stop("A response index exceeds the prediction output dimension.",
         call. = FALSE)
  }
  responses
}

.prediction_to_device_mst_pmdn <- function(pred, device) {
  out <- pred
  for (name in names(out)) {
    if (inherits(out[[name]], "torch_tensor")) {
      out[[name]] <- out[[name]]$to(device = device)
    }
  }
  out
}

.slice_prediction_mst_pmdn <- function(pred, rows) {
  info <- .validate_prediction_mst_pmdn(pred)
  rows <- as.integer(rows)
  if (!length(rows) || anyNA(rows) || any(rows < 1L) ||
      any(rows > info$batch_size)) {
    stop("rows contains invalid 1-based prediction indices.", call. = FALSE)
  }
  out <- pred
  for (name in names(pred)) {
    value <- pred[[name]]
    if (inherits(value, "torch_tensor")) {
      size <- as.integer(value$size())
      if (length(size) >= 1L && size[1] == info$batch_size) {
        index <- torch_tensor(rows, dtype = torch_long(), device = value$device)
        out[[name]] <- torch_index_select(value, 1L, index)
      }
    }
  }
  out
}

.bind_predictions_mst_pmdn <- function(predictions) {
  if (!is.list(predictions) || !length(predictions)) {
    stop("predictions must be a non-empty list.", call. = FALSE)
  }
  lapply(predictions, .validate_prediction_mst_pmdn)
  out <- predictions[[1L]]
  for (name in names(out)) {
    if (inherits(out[[name]], "torch_tensor")) {
      tensors <- lapply(predictions, `[[`, name)
      out[[name]] <- torch_cat(tensors, dim = 1L)
    }
  }
  out
}

.component_moments_mst_pmdn <- function(pred) {
  info <- .validate_prediction_mst_pmdn(pred)
  mu <- pred$mu
  C <- pred$scale_chol$to(device = mu$device, dtype = mu$dtype)
  nu <- pred$nu$to(device = mu$device, dtype = mu$dtype)
  alpha <- pred$alpha$to(device = mu$device, dtype = mu$dtype)
  normal <- nu == Inf

  alpha_norm_sq <- alpha$pow(2)$sum(dim = 3L, keepdim = TRUE)
  delta <- alpha / torch_sqrt(1 + alpha_norm_sq)
  mean_valid <- normal | (nu > 1)
  cov_valid <- normal | (nu > 2)

  nu_mean_safe <- torch_where(
    normal,
    3 * torch_ones_like(nu),
    nu$clamp(min = 1 + 1e-6)
  )
  log_b <- 0.5 * (
    torch_log(nu_mean_safe) -
      torch_log(torch_tensor(pi, dtype = nu$dtype, device = nu$device))
  ) + torch_lgamma((nu_mean_safe - 1) / 2) -
    torch_lgamma(nu_mean_safe / 2)
  b <- torch_where(
    normal,
    sqrt(2 / pi) * torch_ones_like(nu_mean_safe),
    torch_exp(log_b)
  )
  shift <- torch_matmul(
    C,
    (b$unsqueeze(3L) * delta)$unsqueeze(4L)
  )$squeeze(4L)
  component_mean <- mu + shift
  component_mean <- component_mean$masked_fill(
    (!mean_valid)$unsqueeze(3L)$expand(component_mean$size()),
    NaN
  )

  nu_cov_safe <- torch_where(
    normal,
    3 * torch_ones_like(nu),
    nu$clamp(min = 2 + 1e-6)
  )
  log_b_cov <- 0.5 * (
    torch_log(nu_cov_safe) -
      torch_log(torch_tensor(pi, dtype = nu$dtype, device = nu$device))
  ) + torch_lgamma((nu_cov_safe - 1) / 2) -
    torch_lgamma(nu_cov_safe / 2)
  b_sq <- torch_where(
    normal,
    (2 / pi) * torch_ones_like(nu_cov_safe),
    torch_exp(2 * log_b_cov)
  )$unsqueeze(3L)$unsqueeze(4L)
  multiplier <- torch_where(
    normal,
    torch_ones_like(nu_cov_safe),
    nu_cov_safe / (nu_cov_safe - 2)
  )$unsqueeze(3L)$unsqueeze(4L)
  eye <- torch_eye(info$output_dim, dtype = mu$dtype, device = mu$device)$
    unsqueeze(1L)$unsqueeze(1L)$expand(C$size())
  delta_outer <- delta$unsqueeze(4L) * delta$unsqueeze(3L)
  standardized_cov <- multiplier * eye - b_sq * delta_outer
  component_cov <- torch_matmul(
    torch_matmul(C, standardized_cov),
    C$transpose(3L, 4L)
  )
  component_cov <- component_cov$masked_fill(
    (!cov_valid)$unsqueeze(3L)$unsqueeze(4L)$expand(component_cov$size()),
    NaN
  )
  list(
    mean = component_mean,
    covariance = component_cov,
    delta = delta,
    mean_valid = mean_valid,
    covariance_valid = cov_valid
  )
}

.mixture_moments_mst_pmdn <- function(pred) {
  component <- .component_moments_mst_pmdn(pred)
  pi_weight <- pred$pi$unsqueeze(3L)
  mixture_mean <- (pi_weight * component$mean)$sum(dim = 2L)
  centred <- component$mean - mixture_mean$unsqueeze(2L)
  between <- centred$unsqueeze(4L) * centred$unsqueeze(3L)
  mixture_cov <- (
    pred$pi$unsqueeze(3L)$unsqueeze(4L) *
      (component$covariance + between)
  )$sum(dim = 2L)
  list(
    mean = mixture_mean,
    covariance = mixture_cov,
    component = component,
    mean_valid = component$mean_valid$all(dim = 2L),
    covariance_valid = component$covariance_valid$all(dim = 2L)
  )
}

.functional_is_monte_carlo_mst_pmdn <- function(type) {
  type %in% c(
    "quantile", "exceedance", "joint_exceedance",
    "tail_spread", "tail_asymmetry"
  )
}

.analytic_functional_mst_pmdn <- function(pred, functional, responses) {
  type <- functional$type
  info <- .validate_prediction_mst_pmdn(pred)
  if (type %in% c("df", "inverse_df", "skew_direction") &&
      info$n_mixtures != 1L) {
    stop(
      sprintf("type = '%s' is only defined for a one-component prediction.", type),
      call. = FALSE
    )
  }

  if (type == "df") {
    component <- torch_tensor(1L, dtype = torch_long(), device = pred$nu$device)
    nu <- torch_index_select(pred$nu, 2L, component)$squeeze(2L)
    return(as.numeric(torch::as_array(nu$to(device = "cpu"))))
  }
  if (type == "inverse_df") {
    component <- torch_tensor(1L, dtype = torch_long(), device = pred$nu$device)
    nu_tensor <- torch_index_select(pred$nu, 2L, component)$squeeze(2L)
    nu <- as.numeric(torch::as_array(nu_tensor$to(device = "cpu")))
    return(ifelse(is.infinite(nu), 0, 1 / nu))
  }
  if (type == "skew_direction") {
    component <- torch_tensor(
      1L, dtype = torch_long(), device = pred$alpha$device
    )
    alpha <- torch_index_select(pred$alpha, 2L, component)$squeeze(2L)
    delta <- alpha / torch_sqrt(
      1 + alpha$pow(2)$sum(dim = 2L, keepdim = TRUE)
    )
    response <- torch_tensor(
      responses[1L], dtype = torch_long(), device = delta$device
    )
    value <- torch_index_select(delta, 2L, response)$squeeze(2L)
    return(as.numeric(torch::as_array(value$to(device = "cpu"))))
  }

  moments <- .mixture_moments_mst_pmdn(pred)
  if (type == "mean") {
    tensor <- moments$mean[, responses[1L]]
  } else if (type %in% c("variance", "sd")) {
    tensor <- moments$covariance[, responses[1L], responses[1L]]
    if (type == "sd") tensor <- torch_sqrt(tensor)
  } else if (type == "covariance") {
    tensor <- moments$covariance[, responses[1L], responses[2L]]
  } else if (type == "correlation") {
    cov12 <- moments$covariance[, responses[1L], responses[2L]]
    var1 <- moments$covariance[, responses[1L], responses[1L]]
    var2 <- moments$covariance[, responses[2L], responses[2L]]
    tensor <- cov12 / torch_sqrt(var1 * var2)
  } else {
    stop("Internal error: unsupported analytic functional.", call. = FALSE)
  }
  as.numeric(torch::as_array(tensor$to(device = "cpu")))
}

.quantile_columns_mst_pmdn <- function(samples, response, probs) {
  S <- samples$size(1)
  B <- samples$size(2)
  values <- samples[, , response]$to(device = "cpu")
  mat <- matrix(
    as.numeric(torch::as_array(values)),
    nrow = S,
    ncol = B
  )
  if (length(probs) == 1L) {
    probs <- rep(probs, B)
  }
  vapply(
    seq_len(B),
    function(i) stats::quantile(
      mat[, i], probs = probs[i], names = FALSE, type = 7
    ),
    numeric(1)
  )
}

.event_indicator_mst_pmdn <- function(values, threshold, direction) {
  if (direction == "upper") values > threshold else values < threshold
}

.monte_carlo_functional_mst_pmdn <- function(pred,
                                              functional,
                                              responses,
                                              latent_draws,
                                              device) {
  sampled <- .sample_with_latent_mst_pmdn(
    pred, latent_draws = latent_draws, device = device
  )
  samples <- sampled$samples
  S <- samples$size(1)
  B <- samples$size(2)
  type <- functional$type
  if (type == "quantile") {
    return(.quantile_columns_mst_pmdn(
      samples, responses[1L], functional$prob
    ))
  }
  if (type %in% c("tail_spread", "tail_asymmetry")) {
    q_low <- .quantile_columns_mst_pmdn(
      samples, responses[1L], functional$prob[1L]
    )
    q_high <- .quantile_columns_mst_pmdn(
      samples, responses[1L], functional$prob[2L]
    )
    if (type == "tail_spread") return(q_high - q_low)
    q_mid <- .quantile_columns_mst_pmdn(samples, responses[1L], 0.5)
    return(q_high + q_low - 2 * q_mid)
  }

  event1 <- .event_indicator_mst_pmdn(
    samples[, , responses[1L]],
    functional$threshold[1L],
    functional$direction[1L]
  )
  event <- event1
  if (type == "joint_exceedance") {
    event2 <- .event_indicator_mst_pmdn(
      samples[, , responses[2L]],
      functional$threshold[2L],
      functional$direction[2L]
    )
    event <- event1 & event2
  }
  prob <- event$to(dtype = samples$dtype)$mean(dim = 1L)$to(device = "cpu")
  as.numeric(torch::as_array(prob))
}

.apply_functional_transform_mst_pmdn <- function(values,
                                                  functional,
                                                  num_samples = NULL) {
  transform <- functional$transform
  if (transform == "identity") return(values)
  if (transform == "log") {
    invalid <- !is.na(values) & values <= 0
    if (any(invalid)) {
      warning("The log transform is undefined for non-positive functional values.")
      values[invalid] <- NA_real_
    }
    return(log(values))
  }
  if (transform == "logit") {
    eps <- if (is.null(num_samples)) 1e-12 else 0.5 / num_samples
    return(stats::qlogis(pmin(pmax(values, eps), 1 - eps)))
  }
  if (transform == "fisher_z") {
    return(atanh(pmin(pmax(values, -1 + 1e-7), 1 - 1e-7)))
  }
  values
}

.tail_resolution_mst_pmdn <- function(values,
                                       functional,
                                       num_samples) {
  type <- functional$type
  if (type == "quantile") {
    return(rep(num_samples * min(functional$prob, 1 - functional$prob),
               length(values)))
  }
  if (type %in% c("tail_spread", "tail_asymmetry")) {
    return(rep(
      num_samples * min(functional$prob[1L], 1 - functional$prob[2L]),
      length(values)
    ))
  }
  if (type %in% c("exceedance", "joint_exceedance")) {
    probability <- if (functional$transform == "logit") {
      stats::plogis(values)
    } else {
      values
    }
    return(num_samples * pmin(probability, 1 - probability))
  }
  rep(NA_real_, length(values))
}

.auto_functional_chunk_size_mst_pmdn <- function(batch_size,
                                                  num_samples,
                                                  output_dim) {
  max_elements <- 2e6
  max(1L, min(batch_size, as.integer(floor(
    max_elements / max(1, num_samples * output_dim)
  ))))
}

#' Evaluate a scalar MST-PMDN predictive functional
functional_mst_pmdn <- function(pred,
                                functional,
                                num_samples = 4096L,
                                latent_draws = NULL,
                                seed = NULL,
                                chunk_size = NULL,
                                device = "cpu",
                                response_names = NULL,
                                min_tail_draws = 20L) {
  info <- .validate_prediction_mst_pmdn(pred)
  if (!inherits(functional, "mst_functional")) {
    stop("functional must be returned by mst_functional().", call. = FALSE)
  }
  if (is.null(response_names)) response_names <- attr(pred, "response_names")
  responses <- .resolve_responses_mst_pmdn(
    functional, info$output_dim, response_names
  )
  is_mc <- .functional_is_monte_carlo_mst_pmdn(functional$type)
  num_samples <- validate_num_samples(num_samples)
  min_tail_draws <- validate_num_samples(min_tail_draws)

  if (is_mc) {
    if (is.null(latent_draws)) {
      latent_draws <- latent_draws_mst_pmdn(
        num_samples = num_samples,
        output_dim = info$output_dim,
        dtype = pred$mu$dtype,
        device = device,
        seed = seed
      )
    } else {
      latent_draws <- .validate_latent_draws_mst_pmdn(
        latent_draws,
        output_dim = info$output_dim,
        dtype = pred$mu$dtype,
        device = device
      )
      num_samples <- latent_draws$num_samples
    }
  }

  if (is.null(chunk_size)) {
    chunk_size <- if (is_mc) {
      .auto_functional_chunk_size_mst_pmdn(
        info$batch_size, num_samples, info$output_dim
      )
    } else {
      info$batch_size
    }
  } else {
    chunk_size <- validate_num_samples(chunk_size)
  }
  starts <- seq.int(1L, info$batch_size, by = chunk_size)
  values <- numeric(info$batch_size)
  for (start in starts) {
    rows <- seq.int(start, min(info$batch_size, start + chunk_size - 1L))
    chunk <- .slice_prediction_mst_pmdn(pred, rows)
    chunk <- .prediction_to_device_mst_pmdn(chunk, device)
    values[rows] <- if (is_mc) {
      .monte_carlo_functional_mst_pmdn(
        chunk, functional, responses, latent_draws, device
      )
    } else {
      .analytic_functional_mst_pmdn(chunk, functional, responses)
    }
  }
  values <- .apply_functional_transform_mst_pmdn(
    values, functional, if (is_mc) num_samples else NULL
  )

  expected_tail_draws <- if (is_mc) {
    .tail_resolution_mst_pmdn(values, functional, num_samples)
  } else {
    rep(NA_real_, info$batch_size)
  }
  low_resolution <- is_mc & !is.na(expected_tail_draws) &
    expected_tail_draws < min_tail_draws
  if (any(low_resolution)) {
    warning(
      sprintf(
        paste0(
          "Monte Carlo tail resolution is below %d expected draws for %d ",
          "prediction row(s); increase num_samples or inspect diagnostics."
        ),
        min_tail_draws,
        sum(low_resolution)
      ),
      call. = FALSE
    )
  }
  if (!is_mc && any(is.na(values) | !is.finite(values))) {
    warning("One or more analytic functional values are undefined.",
            call. = FALSE)
  }

  data <- data.frame(
    row = seq_len(info$batch_size),
    value = values,
    expected_tail_draws = expected_tail_draws,
    low_tail_resolution = low_resolution
  )
  out <- list(
    data = data,
    functional = functional,
    responses = responses,
    response_names = response_names,
    settings = list(
      method = if (is_mc) "monte_carlo" else "analytic",
      num_samples = if (is_mc) num_samples else NA_integer_,
      chunk_size = chunk_size,
      device = device,
      min_tail_draws = min_tail_draws
    ),
    diagnostics = list(
      low_tail_resolution_rows = which(low_resolution),
      min_expected_tail_draws = if (is_mc) {
        min(expected_tail_draws, na.rm = TRUE)
      } else {
        NA_real_
      }
    ),
    latent_draws = if (is_mc) latent_draws else NULL
  )
  class(out) <- "mst_pmdn_functional"
  out
}

as.data.frame.mst_pmdn_functional <- function(x, ...) x$data

print.mst_pmdn_functional <- function(x, ...) {
  cat(
    "MST-PMDN functional values:\n",
    "  type: ", x$functional$type, "\n",
    "  rows: ", nrow(x$data), "\n",
    "  method: ", x$settings$method, "\n",
    sep = ""
  )
  print(utils::head(x$data), row.names = FALSE)
  invisible(x)
}
