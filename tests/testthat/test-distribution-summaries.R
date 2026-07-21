test_that("marginal CDF is the empirical proportion", {
  pred <- make_mdn_output(
    pi = matrix(1, 1, 1),
    mu = array(0, c(1, 1, 1)),
    scale_chol = array(1, c(1, 1, 1, 1)),
    nu = matrix(10, 1, 1),
    alpha = array(0, c(1, 1, 1))
  )
  draws <- array(seq_len(10), dim = c(10, 1, 1))

  expect_equal(
    as.numeric(cdf_marginal_mst_pmdn(pred, 5, draws = draws)),
    0.5
  )
  expect_equal(
    as.numeric(cdf_marginal_mst_pmdn(pred, 0, draws = draws)),
    0
  )
  expect_equal(
    as.numeric(cdf_marginal_mst_pmdn(pred, 10, draws = draws)),
    1
  )
})

test_that("scov distinguishes scale and covariance", {
  chol <- matrix(c(2, 0, 0.5, 1), 2, 2, byrow = TRUE)
  scale <- chol %*% t(chol)
  pred <- make_mdn_output(
    pi = matrix(1, 1, 1),
    mu = array(0, c(1, 1, 2)),
    scale_chol = array(chol, c(1, 1, 2, 2)),
    nu = matrix(10, 1, 1),
    alpha = array(0, c(1, 1, 2))
  )

  expect_equal(scov_mst_pmdn(pred, as_array = TRUE)[1, 1, , ], scale,
               tolerance = 1e-6)
  expect_equal(
    scov_mst_pmdn(pred, type = "scale_chol", as_array = TRUE)[1, 1, , ],
    chol,
    tolerance = 1e-6
  )
  expect_equal(
    scov_mst_pmdn(pred, type = "cov", as_array = TRUE)[1, 1, , ],
    10 / 8 * scale,
    tolerance = 1e-5
  )
})

test_that("skew-t covariance uses nu and alpha", {
  nu <- 12
  alpha_vec <- c(1, -0.5)
  delta <- alpha_vec / sqrt(1 + sum(alpha_vec^2))
  b_nu <- sqrt(nu / pi) * gamma((nu - 1) / 2) / gamma(nu / 2)
  expected <- nu / (nu - 2) * diag(2) -
    b_nu^2 * tcrossprod(delta)
  pred <- make_mdn_output(
    pi = matrix(1, 1, 1),
    mu = array(0, c(1, 1, 2)),
    scale_chol = array(diag(2), c(1, 1, 2, 2)),
    nu = matrix(nu, 1, 1),
    alpha = array(alpha_vec, c(1, 1, 2))
  )

  expect_equal(
    scov_mst_pmdn(pred, type = "cov", as_array = TRUE)[1, 1, , ],
    expected,
    tolerance = 1e-5
  )

  pred$nu <- torch::torch_tensor(matrix(2, 1, 1))
  expect_warning(
    invalid <- scov_mst_pmdn(pred, type = "cov", as_array = TRUE),
    "undefined"
  )
  expect_true(all(is.nan(invalid)))
})
