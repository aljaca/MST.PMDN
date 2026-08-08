################################################################################
# Mixture-safe exceedance source accounting                                    #
################################################################################

.component_prediction_mst_pmdn <- function(pred, component) {
  info <- .validate_prediction_mst_pmdn(pred)
  if (!is.numeric(component) || length(component) != 1L ||
      component < 1L || component > info$n_mixtures ||
      component != floor(component)) {
    stop("component must be a valid 1-based mixture index.", call. = FALSE)
  }
  component <- as.integer(component)
  index_for <- function(tensor) {
    torch_tensor(component, dtype = torch_long(), device = tensor$device)
  }
  out <- list(
    pi = torch_ones(
      c(info$batch_size, 1L),
      dtype = pred$pi$dtype,
      device = pred$pi$device
    ),
    mu = torch_index_select(pred$mu, 2L, index_for(pred$mu)),
    scale_chol = torch_index_select(
      pred$scale_chol, 2L, index_for(pred$scale_chol)
    ),
    nu = torch_index_select(pred$nu, 2L, index_for(pred$nu)),
    alpha = torch_index_select(pred$alpha, 2L, index_for(pred$alpha)),
    skew_none = .validate_skew_none(pred, "pred")
  )
  out
}

# Attribute a mixture exceedance probability to its components
tail_components_mst_pmdn <- function(pred,
                                     response,
                                     threshold,
                                     direction = "upper",
                                     num_samples = 4096L,
                                     latent_draws = NULL,
                                     seed = NULL,
                                     chunk_size = NULL,
                                     device = "cpu",
                                     response_names = NULL,
                                     min_tail_draws = 20L) {
  info <- .validate_prediction_mst_pmdn(pred)
  functional <- mst_functional(
    type = "exceedance",
    responses = response,
    threshold = threshold,
    direction = direction
  )
  num_samples <- validate_num_samples(num_samples)
  min_tail_draws <- validate_num_samples(min_tail_draws)
  latent_draws <- .ensure_latent_bank_mst_pmdn(
    pred, functional, latent_draws, num_samples, seed, device
  )
  num_samples <- latent_draws$num_samples

  component_probability <- matrix(
    NA_real_, nrow = info$batch_size, ncol = info$n_mixtures
  )
  component_diagnostics <- vector("list", info$n_mixtures)
  for (component in seq_len(info$n_mixtures)) {
    component_pred <- .component_prediction_mst_pmdn(pred, component)
    result <- .functional_values_quiet_mst_pmdn(
      component_pred,
      functional,
      num_samples,
      latent_draws,
      chunk_size,
      device,
      response_names
    )
    component_probability[, component] <- result$data$value
    component_diagnostics[[component]] <- result$diagnostics
  }

  weights <- as.matrix(torch::as_array(pred$pi$to(device = "cpu")))
  contribution <- weights * component_probability
  total <- rowSums(contribution)
  share <- contribution / total
  share[!is.finite(share)] <- NA_real_
  contribution_rank <- matrix(
    NA_integer_, nrow = info$batch_size, ncol = info$n_mixtures
  )
  for (row in seq_len(info$batch_size)) {
    contribution_rank[row, ] <- rank(
      -contribution[row, ], ties.method = "min"
    )
  }
  expected_tail_draws <- num_samples * pmin(total, 1 - total)
  low_resolution <- expected_tail_draws < min_tail_draws
  if (any(low_resolution)) {
    warning(
      sprintf(
        paste0(
          "Mixture exceedance resolution is below %d expected draws for %d ",
          "prediction row(s)."
        ),
        min_tail_draws,
        sum(low_resolution)
      ),
      call. = FALSE
    )
  }

  data <- do.call(rbind, lapply(seq_len(info$batch_size), function(row) {
    data.frame(
      row = row,
      component = seq_len(info$n_mixtures),
      rank = as.integer(contribution_rank[row, ]),
      weight = weights[row, ],
      component_probability = component_probability[row, ],
      contribution = contribution[row, ],
      tail_share = share[row, ],
      total_probability = total[row],
      expected_tail_draws = expected_tail_draws[row],
      low_tail_resolution = low_resolution[row]
    )
  }))
  rownames(data) <- NULL
  out <- list(
    data = data,
    functional = functional,
    settings = list(
      num_samples = num_samples,
      chunk_size = chunk_size,
      device = device,
      min_tail_draws = min_tail_draws
    ),
    diagnostics = list(
      component = component_diagnostics,
      low_tail_resolution_rows = which(low_resolution),
      min_expected_tail_draws = .min_finite_mst_pmdn(
        expected_tail_draws
      )
    ),
    latent_draws = .latent_draws_for_output_mst_pmdn(latent_draws)
  )
  class(out) <- "mst_pmdn_tail_components"
  out
}

as.data.frame.mst_pmdn_tail_components <- function(x, ...) x$data

print.mst_pmdn_tail_components <- function(x, ...) {
  cat(
    "MST-PMDN mixture exceedance contributions:\n",
    "  rows: ", length(unique(x$data$row)), "\n",
    "  components: ", length(unique(x$data$component)), "\n",
    "  Monte Carlo draws: ", x$settings$num_samples, "\n",
    sep = ""
  )
  invisible(x)
}
