responsibilities_mst_pmdn <- function(pred, target, log = FALSE,
                                      as_array = FALSE) {
  info <- .validate_prediction_mst_pmdn(pred)
  if (!is.logical(log) || length(log) != 1L || is.na(log)) {
    stop("log must be a single non-missing logical value.", call. = FALSE)
  }
  if (!is.logical(as_array) || length(as_array) != 1L || is.na(as_array)) {
    stop("as_array must be a single non-missing logical value.", call. = FALSE)
  }

  target_is_tensor <- inherits(target, "torch_tensor")
  if (target_is_tensor && !torch_is_complex(target)) {
    target_size <- as.integer(target$size())
  } else if (is.matrix(target) && is.numeric(target) && !is.complex(target)) {
    target_size <- dim(target)
    target_names <- colnames(target)
    response_names <- attr(pred, "response_names", exact = TRUE)
    if (!is.null(target_names) && !is.null(response_names) &&
        !identical(unname(target_names), unname(response_names))) {
      stop("target column names must match pred's response_names in order.",
           call. = FALSE)
    }
  } else {
    stop("target must be a real numeric matrix or torch tensor.", call. = FALSE)
  }
  if (!identical(target_size, c(info$batch_size, info$output_dim))) {
    stop("target must have one row per prediction and one column per response.",
         call. = FALSE)
  }

  dtype <- pred$mu$dtype
  device <- pred$mu$device
  target <- if (target_is_tensor) {
    target$to(dtype = dtype, device = device)
  } else {
    torch_tensor(target, dtype = dtype, device = device)
  }
  if (!torch_isfinite(target)$all()$item()) {
    stop("target must contain only finite responses.", call. = FALSE)
  }

  weights <- pred$pi
  active <- weights > 0
  if ((!torch_isfinite(weights) | (weights < 0))$any()$item() ||
      !active$any(dim = 2L)$all()$item()) {
    stop("pred$pi must contain finite, non-negative weights with at least one positive weight per row.",
         call. = FALSE)
  }

  components <- .component_log_density_mst_pmdn(
    pred, target, skew_none = isTRUE(pred$skew_none)
  )
  log_density <- components$log_density$masked_fill(!active, -Inf)
  max_log_density <- torch_amax(log_density, dim = 2L, keepdim = TRUE)
  if (!torch_isfinite(max_log_density)$all()$item()) {
    stop("Each row must have a finite maximum active component log density.",
         call. = FALSE)
  }

  # Center before adding weights to preserve their precision in the tails.
  # This shared shift cancels in the softmax and its derivatives.
  log_density <- log_density - max_log_density$detach()
  # Replace zeros before log so autograd never differentiates through log(0).
  log_weights <- torch_where(active, weights, torch_ones_like(weights))$log()
  log_scores <- log_density + log_weights
  result <- if (log) {
    nnf_log_softmax(log_scores, dim = 2L)
  } else {
    nnf_softmax(log_scores, dim = 2L)
  }
  if (as_array) {
    return(torch::as_array(result$detach()$to(device = "cpu")))
  }
  result
}
