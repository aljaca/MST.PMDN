make_channel_pair <- function(mu_from = 0, mu_to = mu_from,
                              scale_from = 1, scale_to = scale_from,
                              alpha_from = 0, alpha_to = alpha_from,
                              nu_from = Inf, nu_to = nu_from,
                              skew_none = FALSE) {
  list(
    from = make_mdn_output(
      pi = matrix(1, 1, 1),
      mu = array(mu_from, c(1, 1, 1)),
      scale_chol = array(scale_from, c(1, 1, 1, 1)),
      nu = matrix(nu_from, 1, 1),
      alpha = array(alpha_from, c(1, 1, 1)),
      skew_none = skew_none
    ),
    to = make_mdn_output(
      pi = matrix(1, 1, 1),
      mu = array(mu_to, c(1, 1, 1)),
      scale_chol = array(scale_to, c(1, 1, 1, 1)),
      nu = matrix(nu_to, 1, 1),
      alpha = array(alpha_to, c(1, 1, 1)),
      skew_none = skew_none
    )
  )
}

test_that("single active channels receive the complete contrast", {
  cases <- list(
    location = list(
      pair = make_channel_pair(mu_to = 2),
      functional = mst_functional("mean", 1L)
    ),
    scale = list(
      pair = make_channel_pair(scale_to = 2, skew_none = TRUE),
      functional = mst_functional("sd", 1L)
    ),
    skewness = list(
      pair = make_channel_pair(alpha_to = 1),
      functional = mst_functional("mean", 1L)
    ),
    df = list(
      pair = make_channel_pair(
        nu_from = 5, nu_to = 10, skew_none = TRUE
      ),
      functional = mst_functional("variance", 1L)
    )
  )
  for (channel in names(cases)) {
    item <- cases[[channel]]
    result <- decompose_mst_pmdn(
      item$pair$from,
      item$pair$to,
      item$functional
    )
    expect_identical(result$active_channels, channel)
    expect_equal(
      result$data[[paste0("channel_", channel)]],
      result$data$total,
      tolerance = 1e-6
    )
    expect_equal(result$data$sum_to_total_residual, 0, tolerance = 1e-10)
  }
})

test_that("Shapley channel contributions close under interacting changes", {
  pair <- make_channel_pair(
    mu_to = 0.7,
    scale_to = 1.4,
    alpha_from = -0.4,
    alpha_to = 0.8,
    nu_from = 6,
    nu_to = 15
  )
  bank <- latent_draws_mst_pmdn(4096L, output_dim = 1L, seed = 31)
  result <- suppressWarnings(decompose_mst_pmdn(
    pair$from,
    pair$to,
    mst_functional("quantile", 1L, prob = 0.95),
    latent_draws = bank
  ))
  expect_setequal(
    result$active_channels,
    c("location", "scale", "skewness", "df")
  )
  expect_equal(result$data$sum_to_total_residual, 0, tolerance = 1e-10)
  expect_false(".cache" %in% names(result$latent_draws))
})

test_that("structurally inactive skewness and df channels disappear", {
  pair <- make_channel_pair(mu_to = 1, skew_none = TRUE)
  result <- decompose_mst_pmdn(
    pair$from,
    pair$to,
    mst_functional("mean", 1L)
  )
  expect_identical(result$active_channels, "location")
  expect_false("channel_skewness" %in% names(result$data))
  expect_false("channel_df" %in% names(result$data))
})

test_that("full parameter-channel decomposition rejects mixtures", {
  pred <- make_mdn_output(
    pi = matrix(c(0.5, 0.5), 1, 2),
    mu = array(c(-1, 1), c(1, 2, 1)),
    scale_chol = array(1, c(1, 2, 1, 1)),
    nu = matrix(Inf, 1, 2),
    alpha = array(0, c(1, 2, 1)),
    skew_none = TRUE
  )
  expect_error(
    decompose_mst_pmdn(
      pred, pred, mst_functional("mean", 1L)
    ),
    "only available for M = 1"
  )
})

test_that("structural and exact numerical symmetry are equivalent", {
  pair <- make_channel_pair(skew_none = TRUE)
  pair$from$skew_none <- FALSE
  result <- decompose_mst_pmdn(
    pair$from,
    pair$to,
    mst_functional("mean", 1L)
  )
  expect_length(result$active_channels, 0L)
  expect_equal(result$data$total, 0, tolerance = 0)
})

test_that("decomposition reports exact parameter change magnitudes", {
  pair <- make_channel_pair(
    mu_to = 2,
    scale_to = 3,
    alpha_from = -0.5,
    alpha_to = 1,
    nu_from = 5,
    nu_to = 10
  )
  result <- decompose_mst_pmdn(
    pair$from,
    pair$to,
    mst_functional("mean", 1L)
  )
  expect_equal(
    result$diagnostics$max_abs_parameter_change,
    c(location = 2, scale = 2, skewness = 1.5, df = 0.1),
    tolerance = 1e-7
  )
})

test_that("chunk-first Shapley evaluation reuses each df parent", {
  B <- 20L
  pred_from <- make_mdn_output(
    pi = matrix(1, B, 1),
    mu = array(seq(-0.5, 0.5, length.out = B), c(B, 1, 1)),
    scale_chol = array(1, c(B, 1, 1, 1)),
    nu = matrix(6, B, 1),
    alpha = array(-0.4, c(B, 1, 1))
  )
  pred_to <- make_mdn_output(
    pi = matrix(1, B, 1),
    mu = array(seq(0.2, 1.2, length.out = B), c(B, 1, 1)),
    scale_chol = array(1.4, c(B, 1, 1, 1)),
    nu = matrix(15, B, 1),
    alpha = array(0.8, c(B, 1, 1))
  )
  base_bank <- latent_draws_mst_pmdn(
    128L, output_dim = 1L, seed = 32
  )
  chunk_bank <- base_bank
  chunk_bank$.cache <- new.env(parent = emptyenv())
  whole_bank <- base_bank
  whole_bank$.cache <- new.env(parent = emptyenv())
  functional <- mst_functional("quantile", 1L, prob = 0.9)

  chunked <- suppressWarnings(decompose_mst_pmdn(
    pred_from,
    pred_to,
    functional,
    latent_draws = chunk_bank,
    chunk_size = 2L
  ))
  whole <- suppressWarnings(decompose_mst_pmdn(
    pred_from,
    pred_to,
    functional,
    latent_draws = whole_bank,
    chunk_size = B
  ))

  expect_equal(chunked$data, whole$data, tolerance = 0)
  expect_equal(chunked$settings$chunks, 10L)
  expect_equal(chunk_bank$.cache$gamma_scale_misses, 20L)
  expect_equal(chunk_bank$.cache$gamma_scale_hits, 140L)
  expect_true(length(chunk_bank$.cache$gamma_scale_entries) <= 8L)
})

test_that("decomposition aggregates a rare-tail warning once", {
  pair <- make_channel_pair(mu_to = 1)
  bank <- latent_draws_mst_pmdn(64L, output_dim = 1L, seed = 33)
  warning_count <- 0L
  result <- withCallingHandlers(
    decompose_mst_pmdn(
      pair$from,
      pair$to,
      mst_functional("exceedance", 1L, threshold = 1e30),
      latent_draws = bank,
      min_tail_draws = 20L
    ),
    mst_pmdn_tail_resolution_warning = function(condition) {
      warning_count <<- warning_count + 1L
      invokeRestart("muffleWarning")
    }
  )
  expect_equal(warning_count, 1L)
  expect_equal(result$diagnostics$min_expected_tail_draws, 0)
  expect_gt(result$diagnostics$low_tail_resolution_evaluations, 0)
})
