test_that("exact Gaussian and Student-t moments are returned", {
  gaussian <- make_mdn_output(
    pi = matrix(1, 1, 1),
    mu = array(2, c(1, 1, 1)),
    scale_chol = array(3, c(1, 1, 1, 1)),
    nu = matrix(Inf, 1, 1),
    alpha = array(0, c(1, 1, 1)),
    skew_none = TRUE
  )
  student <- gaussian
  student$nu <- torch::torch_tensor(matrix(8, 1, 1))

  expect_equal(
    functional_mst_pmdn(
      gaussian, mst_functional("mean", 1L)
    )$data$value,
    2
  )
  expect_equal(
    functional_mst_pmdn(
      gaussian, mst_functional("variance", 1L)
    )$data$value,
    9,
    tolerance = 1e-6
  )
  expect_equal(
    functional_mst_pmdn(
      student, mst_functional("variance", 1L)
    )$data$value,
    8 / 6 * 9,
    tolerance = 1e-5
  )
  expect_warning(
    exact_df <- functional_mst_pmdn(
      gaussian, mst_functional("df")
    ),
    NA
  )
  expect_identical(exact_df$data$value, Inf)
})

test_that("exact skew-normal and skew-t moments are returned", {
  alpha <- 1.25
  delta <- alpha / sqrt(1 + alpha^2)
  for (nu in c(Inf, 12)) {
    pred <- make_mdn_output(
      pi = matrix(1, 1, 1),
      mu = array(0.5, c(1, 1, 1)),
      scale_chol = array(2, c(1, 1, 1, 1)),
      nu = matrix(nu, 1, 1),
      alpha = array(alpha, c(1, 1, 1))
    )
    if (is.infinite(nu)) {
      b <- sqrt(2 / pi)
      multiplier <- 1
    } else {
      b <- sqrt(nu / pi) * gamma((nu - 1) / 2) / gamma(nu / 2)
      multiplier <- nu / (nu - 2)
    }
    expected_mean <- 0.5 + 2 * b * delta
    expected_variance <- 4 * (multiplier - b^2 * delta^2)
    expect_equal(
      functional_mst_pmdn(
        pred, mst_functional("mean", 1L)
      )$data$value,
      expected_mean,
      tolerance = 2e-5
    )
    expect_equal(
      functional_mst_pmdn(
        pred, mst_functional("variance", 1L)
      )$data$value,
      expected_variance,
      tolerance = 2e-5
    )
  }
})

test_that("mixture covariance contains between-component variation", {
  weights <- c(0.25, 0.75)
  means <- c(-1, 2)
  pred <- make_mdn_output(
    pi = matrix(weights, 1, 2),
    mu = array(means, c(1, 2, 1)),
    scale_chol = array(1, c(1, 2, 1, 1)),
    nu = matrix(Inf, 1, 2),
    alpha = array(0, c(1, 2, 1)),
    skew_none = TRUE
  )
  expected_mean <- sum(weights * means)
  expected_variance <- sum(weights * (1 + (means - expected_mean)^2))
  expect_equal(
    functional_mst_pmdn(pred, mst_functional("mean", 1L))$data$value,
    expected_mean,
    tolerance = 1e-6
  )
  expect_equal(
    functional_mst_pmdn(
      pred, mst_functional("variance", 1L)
    )$data$value,
    expected_variance,
    tolerance = 1e-6
  )
})

test_that("undefined low-df moments are explicit", {
  pred <- make_mdn_output(
    pi = matrix(1, 1, 1),
    mu = array(0, c(1, 1, 1)),
    scale_chol = array(1, c(1, 1, 1, 1)),
    nu = matrix(0.9, 1, 1),
    alpha = array(0.5, c(1, 1, 1))
  )
  mean_value <- suppressWarnings(functional_mst_pmdn(
    pred, mst_functional("mean", 1L)
  )$data$value)
  expect_true(is.nan(mean_value) || is.na(mean_value))

  pred$nu <- torch::torch_tensor(matrix(2, 1, 1))
  variance <- suppressWarnings(functional_mst_pmdn(
    pred, mst_functional("variance", 1L)
  )$data$value)
  expect_true(is.nan(variance) || is.na(variance))
})

test_that("functional response names and pairwise selectors are validated", {
  pred <- make_mdn_output(
    pi = matrix(1, 1, 1),
    mu = array(c(0, 0), c(1, 1, 2)),
    scale_chol = array(diag(2), c(1, 1, 2, 2)),
    nu = matrix(Inf, 1, 1),
    alpha = array(0, c(1, 1, 2)),
    skew_none = TRUE
  )
  attr(pred, "response_names") <- c("wave", "surge")
  correlation <- functional_mst_pmdn(
    pred,
    mst_functional("correlation", c("wave", "surge"))
  )
  expect_equal(correlation$data$value, 0, tolerance = 1e-6)
  expect_error(mst_functional("mean", c(1L, 2L)), "length 1")
  expect_error(mst_functional("correlation", 1L), "length 2")
})

test_that("tail diagnostics expose inadequate quantile resolution", {
  pred <- make_mdn_output(
    pi = matrix(1, 1, 1),
    mu = array(0, c(1, 1, 1)),
    scale_chol = array(1, c(1, 1, 1, 1)),
    nu = matrix(Inf, 1, 1),
    alpha = array(0, c(1, 1, 1)),
    skew_none = TRUE
  )
  expect_warning(
    result <- functional_mst_pmdn(
      pred,
      mst_functional("quantile", 1L, prob = 0.999),
      num_samples = 4096L,
      seed = 1
    ),
    "tail resolution"
  )
  expect_equal(result$data$expected_tail_draws, 4.096, tolerance = 1e-10)
  expect_true(result$data$low_tail_resolution)
})

test_that("covariance uses the complete Cholesky scale factor", {
  chol <- array(c(2, 1, 0, 3), c(1, 1, 2, 2))
  pred <- make_mdn_output(
    pi = matrix(1, 1, 1),
    mu = array(0, c(1, 1, 2)),
    scale_chol = chol,
    nu = matrix(Inf, 1, 1),
    alpha = array(0, c(1, 1, 2)),
    skew_none = TRUE
  )
  result <- functional_mst_pmdn(
    pred, mst_functional("covariance", c(1L, 2L))
  )
  expect_equal(result$data$value, 2, tolerance = 1e-6)
})

test_that("joint exceedance matches an independent Gaussian probability", {
  pred <- make_mdn_output(
    pi = matrix(1, 1, 1),
    mu = array(0, c(1, 1, 2)),
    scale_chol = array(diag(2), c(1, 1, 2, 2)),
    nu = matrix(Inf, 1, 1),
    alpha = array(0, c(1, 1, 2)),
    skew_none = TRUE
  )
  bank <- latent_draws_mst_pmdn(4L, output_dim = 2L, seed = 51)
  bank$skew_z <- torch::torch_tensor(
    matrix(c(-1, -1, -1, 1, 1, -1, 1, 1), ncol = 2, byrow = TRUE),
    dtype = pred$mu$dtype
  )
  result <- suppressWarnings(functional_mst_pmdn(
    pred,
    mst_functional(
      "joint_exceedance", c(1L, 2L), threshold = c(0, 0)
    ),
    latent_draws = bank,
    min_tail_draws = 1L
  ))
  expect_equal(result$data$value, 0.25, tolerance = 0)
})

test_that("tail spread and normalized asymmetry have distinct scale behavior", {
  pred <- make_mdn_output(
    pi = matrix(1, 2, 1),
    mu = array(c(0, 5), c(2, 1, 1)),
    scale_chol = array(c(1, 3), c(2, 1, 1, 1)),
    nu = matrix(Inf, 2, 1),
    alpha = array(0, c(2, 1, 1)),
    skew_none = TRUE
  )
  bank <- latent_draws_mst_pmdn(5L, output_dim = 1L, seed = 52)
  bank$skew_z <- torch::torch_tensor(
    matrix(c(-3, -1, 0, 2, 4), ncol = 1),
    dtype = pred$mu$dtype
  )
  spread <- suppressWarnings(functional_mst_pmdn(
    pred,
    mst_functional("tail_spread", 1L, prob = c(0.25, 0.75)),
    latent_draws = bank,
    min_tail_draws = 1L
  ))
  asymmetry <- suppressWarnings(functional_mst_pmdn(
    pred,
    mst_functional("tail_asymmetry", 1L, prob = c(0.25, 0.75)),
    latent_draws = bank,
    min_tail_draws = 1L
  ))
  expect_equal(spread$data$value, c(3, 9), tolerance = 1e-6)
  expect_equal(asymmetry$data$value, rep(1 / 3, 2), tolerance = 1e-6)
})

test_that("inverse df and bounded skew direction are explicit diagnostics", {
  alpha <- array(0, c(2, 1, 2))
  alpha[1, 1, 2] <- 3
  scale_chol <- array(0, c(2, 1, 2, 2))
  scale_chol[, 1, 1, 1] <- 1
  scale_chol[, 1, 2, 2] <- 1
  pred <- make_mdn_output(
    pi = matrix(1, 2, 1),
    mu = array(0, c(2, 1, 2)),
    scale_chol = scale_chol,
    nu = matrix(c(Inf, 4), 2, 1),
    alpha = alpha
  )
  inverse_df <- functional_mst_pmdn(
    pred, mst_functional("inverse_df")
  )
  skew_direction <- functional_mst_pmdn(
    pred, mst_functional("skew_direction", 2L)
  )
  expect_equal(inverse_df$data$value, c(0, 0.25), tolerance = 1e-7)
  expect_equal(
    skew_direction$data$value,
    c(3 / sqrt(10), 0),
    tolerance = 1e-6
  )
})

test_that("mixture diagnostics expose expected component draw counts", {
  pred <- make_mdn_output(
    pi = matrix(c(0.99, 0.01), 1, 2),
    mu = array(c(-1, 1), c(1, 2, 1)),
    scale_chol = array(1, c(1, 2, 1, 1)),
    nu = matrix(Inf, 1, 2),
    alpha = array(0, c(1, 2, 1)),
    skew_none = TRUE
  )
  result <- suppressWarnings(functional_mst_pmdn(
    pred,
    mst_functional("exceedance", 1L, threshold = 0),
    num_samples = 100L,
    seed = 53,
    min_tail_draws = 1L
  ))
  expect_equal(
    as.numeric(result$diagnostics$expected_component_draws),
    c(99, 1),
    tolerance = 1e-6
  )
  expect_equal(
    result$diagnostics$min_expected_component_draws,
    1,
    tolerance = 1e-6
  )
  expect_true(result$diagnostics$component_draws_shared_across_rows)
})

test_that("automatic functional chunks budget the quadratic scale gather", {
  univariate <- MST.PMDN:::.auto_functional_chunk_size_mst_pmdn(
    batch_size = 1000L, num_samples = 100L, output_dim = 1L
  )
  multivariate <- MST.PMDN:::.auto_functional_chunk_size_mst_pmdn(
    batch_size = 1000L, num_samples = 100L, output_dim = 10L
  )
  expect_equal(univariate, 1000L)
  expect_equal(multivariate, 200L)
})

test_that("probability transforms do not alter tail-resolution diagnostics", {
  pred <- make_mdn_output(
    pi = matrix(1, 1, 1),
    mu = array(0, c(1, 1, 1)),
    scale_chol = array(1, c(1, 1, 1, 1)),
    nu = matrix(Inf, 1, 1),
    alpha = array(0, c(1, 1, 1)),
    skew_none = TRUE
  )
  expect_warning(
    result <- functional_mst_pmdn(
      pred,
      mst_functional(
        "exceedance",
        1L,
        threshold = 100,
        transform = "logit"
      ),
      num_samples = 64L,
      seed = 2
    ),
    "tail resolution"
  )
  expect_equal(result$data$expected_tail_draws, 0)
  expect_equal(result$data$value, stats::qlogis(0.5 / 64))
})


test_that("rare joint exceedance reports its exact Monte Carlo resolution", {
  pred <- make_mdn_output(
    pi = matrix(1, 1, 1),
    mu = array(0, c(1, 1, 2)),
    scale_chol = array(diag(2), c(1, 1, 2, 2)),
    nu = matrix(Inf, 1, 1),
    alpha = array(0, c(1, 1, 2)),
    skew_none = TRUE
  )
  bank <- latent_draws_mst_pmdn(64L, output_dim = 2L, seed = 81)
  z <- matrix(0, nrow = 64L, ncol = 2L)
  z[1L, ] <- 3
  bank$skew_z <- torch::torch_tensor(z, dtype = pred$mu$dtype)

  expect_warning(
    result <- functional_mst_pmdn(
      pred,
      mst_functional(
        "joint_exceedance",
        responses = c(1L, 2L),
        threshold = c(2, 2)
      ),
      latent_draws = bank,
      min_tail_draws = 2L
    ),
    class = "mst_pmdn_tail_resolution_warning"
  )
  expect_equal(result$data$value, 1 / 64, tolerance = 0)
  expect_equal(result$data$expected_tail_draws, 1, tolerance = 0)
  expect_identical(result$diagnostics$low_tail_resolution_count, 1L)
  expect_identical(result$diagnostics$tail_resolution_evaluations, 1L)
})
