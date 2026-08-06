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
