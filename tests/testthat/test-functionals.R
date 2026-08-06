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
