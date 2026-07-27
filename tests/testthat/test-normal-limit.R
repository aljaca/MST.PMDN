normal_test_values <- function(x) {
  as.numeric(torch::as_array(x$to(device = "cpu")))
}

test_that("normal constraints and fixed Inf use the exact limit", {
  x <- torch::torch_zeros(c(4, 2))

  normal_model <- define_mst_pmdn(
    input_dim = 2,
    output_dim = 1,
    hidden_dim = c(3),
    n_mixtures = 2,
    constraint = "VIINN"
  )
  normal_pred <- normal_model(x)
  expect_true(all(is.infinite(normal_test_values(normal_pred$nu))))

  mixed_model <- define_mst_pmdn(
    input_dim = 2,
    output_dim = 1,
    hidden_dim = c(3),
    n_mixtures = 2,
    constraint = "VIIFN",
    constant_attr = "n",
    fixed_nu = c(Inf, NA)
  )
  mixed_nu <- torch::as_array(mixed_model(x)$nu)
  expect_true(all(is.infinite(mixed_nu[, 1])))
  expect_equal(mixed_nu[, 2], rep(26.5, 4), tolerance = 1e-6)
})

test_that("normal and mixed fixed nu paths preserve float64 dtype", {
  x <- torch::torch_zeros(c(4, 2), dtype = torch::torch_double())
  normal_model <- define_mst_pmdn(
    input_dim = 2,
    output_dim = 2,
    hidden_dim = c(3),
    n_mixtures = 2,
    constraint = "VVINN"
  )
  normal_model <- normal_model$to(dtype = torch::torch_double())
  normal_pred <- normal_model(x)
  expect_identical(as.character(normal_pred$nu$dtype), "Double")

  model <- define_mst_pmdn(
    input_dim = 2,
    output_dim = 2,
    hidden_dim = c(3),
    n_mixtures = 2,
    constraint = "VVIFN",
    constant_attr = "n",
    fixed_nu = c(Inf, NA)
  )
  model <- model$to(dtype = torch::torch_double())
  target <- torch::torch_zeros(c(4, 2), dtype = torch::torch_double())

  pred <- model(x)
  for (name in c("pi", "mu", "scale_chol", "nu", "alpha", "L", "A", "D")) {
    expect_identical(as.character(pred[[name]]$dtype), "Double", info = name)
  }

  loss <- loss_mst_pmdn(pred, target)
  expect_identical(as.character(loss$dtype), "Double")
  expect_true(is.finite(loss$item()))
  loss$backward()
  expect_true(all(is.finite(normal_test_values(model$nu_param_partial$grad))))
})

test_that("degrees-of-freedom inputs are validated", {
  expect_no_error(
    define_mst_pmdn(
      input_dim = 1,
      output_dim = 1,
      hidden_dim = c(2),
      n_mixtures = 2,
      constraint = "VIIFN",
      fixed_nu = c(NA, NA)
    )
  )
  expect_error(
    define_mst_pmdn(
      input_dim = 1,
      output_dim = 1,
      hidden_dim = c(2),
      n_mixtures = 1,
      range_nu = c(50, 3)
    ),
    "increasing"
  )
  expect_error(
    define_mst_pmdn(
      input_dim = 1,
      output_dim = 1,
      hidden_dim = c(2),
      n_mixtures = 2,
      constraint = "VIIFN",
      fixed_nu = c(-Inf, NA)
    ),
    "positive finite"
  )
})

test_that("exact Gaussian loss matches the analytic density", {
  target_values <- rbind(c(0, 0), c(1, -2), c(0.5, 0.25))
  scale_chol <- array(0, c(3, 1, 2, 2))
  scale_chol[, , 1, 1] <- 1
  scale_chol[, , 2, 2] <- 1
  pred <- make_mdn_output(
    pi = matrix(1, 3, 1),
    mu = array(0, c(3, 1, 2)),
    scale_chol = scale_chol,
    nu = matrix(Inf, 3, 1),
    alpha = array(0, c(3, 1, 2))
  )
  target <- torch::torch_tensor(
    target_values,
    dtype = torch::torch_float()
  )
  expected <- mean(log(2 * pi) + 0.5 * rowSums(target_values^2))

  expect_equal(
    loss_mst_pmdn(pred, target, lambda_nu_inv = 1)$item(),
    expected,
    tolerance = 1e-6
  )
})

test_that("exact skew-normal loss matches the analytic density", {
  target_values <- matrix(c(-2, 0, 1.5), ncol = 1)
  alpha_value <- 1.25
  pred <- make_mdn_output(
    pi = matrix(1, 3, 1),
    mu = array(0, c(3, 1, 1)),
    scale_chol = array(1, c(3, 1, 1, 1)),
    nu = matrix(Inf, 3, 1),
    alpha = array(alpha_value, c(3, 1, 1))
  )
  target <- torch::torch_tensor(
    target_values,
    dtype = torch::torch_float()
  )
  expected <- -mean(
    log(2) +
    stats::dnorm(target_values[, 1], log = TRUE) +
    stats::pnorm(alpha_value * target_values[, 1], log.p = TRUE)
  )

  expect_equal(loss_mst_pmdn(pred, target)$item(), expected,
               tolerance = 1e-6)
})

test_that("mixed exact-normal and finite-t loss branches component-wise", {
  target_values <- c(-1, 2)
  pred <- make_mdn_output(
    pi = matrix(0.5, 2, 2),
    mu = array(0, c(2, 2, 1)),
    scale_chol = array(1, c(2, 2, 1, 1)),
    nu = matrix(c(Inf, 10, Inf, 10), 2, 2, byrow = TRUE),
    alpha = array(0, c(2, 2, 1))
  )
  target <- torch::torch_tensor(
    matrix(target_values, ncol = 1),
    dtype = torch::torch_float()
  )
  expected <- -mean(log(
    0.5 * stats::dnorm(target_values) +
    0.5 * stats::dt(target_values, df = 10)
  ))

  expect_equal(loss_mst_pmdn(pred, target)$item(), expected,
               tolerance = 2e-6)
})

test_that("bivariate t normalizer removes float32 gamma cancellation", {
  nu <- torch::torch_tensor(
    c(30, 50, 500),
    dtype = torch::torch_float(),
    requires_grad = TRUE
  )
  log_constant <- MST.PMDN:::.log_multivariate_t_normalizer(nu, 2)

  expect_equal(
    normal_test_values(log_constant),
    rep(-log(2 * pi), 3),
    tolerance = 1e-7
  )
  log_constant$sum()$backward()
  expect_equal(normal_test_values(nu$grad), rep(0, 3), tolerance = 1e-8)
})

test_that("stable t normalizer agrees with the analytic constant", {
  nu_values <- c(3, 10, 50)
  for (d in 1:5) {
    actual <- MST.PMDN:::.log_multivariate_t_normalizer(
      torch::torch_tensor(nu_values, dtype = torch::torch_float()),
      d
    )
    expected <- lgamma((nu_values + d) / 2) -
                lgamma(nu_values / 2) -
                (d / 2) * log(nu_values * pi)
    expect_equal(normal_test_values(actual), expected, tolerance = 2e-6,
                 info = paste("dimension", d))
  }
})

test_that("normal covariance uses the exact skew-normal limit", {
  alpha_vec <- c(1, -0.5)
  delta <- alpha_vec / sqrt(1 + sum(alpha_vec^2))
  expected <- diag(2) - (2 / pi) * tcrossprod(delta)
  pred <- make_mdn_output(
    pi = matrix(1, 1, 1),
    mu = array(0, c(1, 1, 2)),
    scale_chol = array(diag(2), c(1, 1, 2, 2)),
    nu = matrix(Inf, 1, 1),
    alpha = array(alpha_vec, c(1, 1, 2))
  )

  expect_equal(
    scov_mst_pmdn(pred, type = "cov", as_array = TRUE)[1, 1, , ],
    expected,
    tolerance = 1e-6
  )
})

test_that("normal sampling omits Student-t scale variability", {
  skip_on_cran()
  pred <- make_mdn_output(
    pi = matrix(1, 1, 1),
    mu = array(0, c(1, 1, 1)),
    scale_chol = array(1, c(1, 1, 1, 1)),
    nu = matrix(Inf, 1, 1),
    alpha = array(0, c(1, 1, 1))
  )

  set.seed(17)
  torch::torch_manual_seed(17)
  draws <- normal_test_values(
    sample_mst_pmdn(pred, num_samples = 20000)$samples
  )

  expect_equal(mean(draws), 0, tolerance = 0.025)
  expect_equal(stats::var(draws), 1, tolerance = 0.04)
})
