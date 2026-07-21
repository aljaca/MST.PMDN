test_that("mixture sampling uses one-based component indices exactly once", {
  B <- 2L
  M <- 2L
  d <- 1L
  pi <- rbind(c(1, 0), c(0, 1))
  mu <- array(0, dim = c(B, M, d))
  mu[, 1, 1] <- -2
  mu[, 2, 1] <- 2
  scale_chol <- array(0, dim = c(B, M, d, d))
  scale_chol[, , 1, 1] <- 1
  nu <- matrix(20, nrow = B, ncol = M)
  alpha <- array(0, dim = c(B, M, d))
  pred <- make_mdn_output(pi, mu, scale_chol, nu, alpha)

  set.seed(1)
  torch::torch_manual_seed(1)
  sampled <- sample_mst_pmdn(pred, num_samples = 20)
  components <- torch::as_array(sampled$components)

  expect_true(all(components[, 1] == 1))
  expect_true(all(components[, 2] == 2))
  expect_true(all(components >= 1 & components <= M))
})

test_that("one-component sampling and data-frame formatting work", {
  pred <- make_mdn_output(
    pi = matrix(1, 1, 1),
    mu = array(0, c(1, 1, 1)),
    scale_chol = array(1, c(1, 1, 1, 1)),
    nu = matrix(10, 1, 1),
    alpha = array(0, c(1, 1, 1))
  )

  expect_no_error(sample_mst_pmdn(pred, num_samples = 3))
  out <- sample_mst_pmdn_df(pred, num_samples = 3)
  expect_equal(nrow(out), 3)
  expect_equal(as.integer(out$comp), rep(1L, 3))
  expect_error(sample_mst_pmdn(pred, num_samples = 1.5),
               "positive integer")
})

test_that("multivariate skew-t samples match analytic moments", {
  skip_on_cran()
  nu <- 20
  alpha_vec <- c(1, 1)
  delta <- alpha_vec / sqrt(1 + sum(alpha_vec^2))
  b_nu <- sqrt(nu / pi) * gamma((nu - 1) / 2) / gamma(nu / 2)
  expected_mean <- b_nu * delta
  expected_cov <- nu / (nu - 2) * diag(2) -
    b_nu^2 * tcrossprod(delta)
  pred <- make_mdn_output(
    pi = matrix(1, 1, 1),
    mu = array(0, c(1, 1, 2)),
    scale_chol = array(diag(2), c(1, 1, 2, 2)),
    nu = matrix(nu, 1, 1),
    alpha = array(alpha_vec, c(1, 1, 2))
  )

  set.seed(42)
  torch::torch_manual_seed(42)
  draws <- torch::as_array(
    sample_mst_pmdn(pred, num_samples = 30000)$samples
  )[, 1, ]

  expect_equal(colMeans(draws), expected_mean, tolerance = 0.035)
  expect_equal(stats::cov(draws), expected_cov, tolerance = 0.05)
})
