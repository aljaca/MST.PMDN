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
  expect_identical(normal_pred$skew_none, TRUE)
  expect_true(all(normal_test_values(normal_pred$alpha) == 0))
  expect_null(normal_model$fc_alpha)
  expect_null(normal_model$alpha_param)

  shared_skew_model <- define_mst_pmdn(
    input_dim = 2,
    output_dim = 1,
    hidden_dim = c(3),
    n_mixtures = 2,
    constraint = "VIINE"
  )
  varying_skew_model <- define_mst_pmdn(
    input_dim = 2,
    output_dim = 1,
    hidden_dim = c(3),
    n_mixtures = 2,
    constraint = "VIINV"
  )
  expect_identical(shared_skew_model(x)$skew_none, FALSE)
  expect_identical(varying_skew_model(x)$skew_none, FALSE)

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

test_that("constant skewness remains on the general skew path", {
  torch::torch_manual_seed(419)
  x <- torch::torch_zeros(c(4, 2))
  model <- define_mst_pmdn(
    input_dim = 2,
    output_dim = 2,
    hidden_dim = c(3),
    n_mixtures = 2,
    constraint = "VIINE",
    constant_attr = "s"
  )
  pred <- model(x)
  alpha <- torch::as_array(pred$alpha)

  expect_identical(pred$skew_none, FALSE)
  expect_null(model$fc_alpha)
  expect_false(is.null(model$alpha_param))
  expect_gt(max(abs(alpha)), 0)
  for (b in seq_len(dim(alpha)[1])) {
    for (m in seq_len(dim(alpha)[2])) {
      expect_equal(alpha[b, m, ], alpha[1, 1, ], tolerance = 0)
    }
  }
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

test_that("mixed float64 loss and gradients agree with analytic references", {
  y <- 0.4
  weights <- c(0.65, 0.35)
  locations <- c(-0.5, 0.75)
  scales <- c(0.8, 1.4)
  alpha <- c(0.3, -0.4)
  nu <- c(Inf, 1)
  lambda_alpha <- 0.2
  lambda_nu_inv <- 0.3
  pred <- make_mdn_output(
    pi = matrix(weights, 1),
    mu = array(locations, c(1, 2, 1)),
    scale_chol = array(scales, c(1, 2, 1, 1)),
    nu = matrix(nu, 1),
    alpha = array(alpha, c(1, 2, 1)),
    dtype = torch::torch_double()
  )
  for (name in c("pi", "mu", "scale_chol", "alpha")) {
    pred[[name]]$requires_grad_(TRUE)
  }
  target <- torch::torch_tensor(matrix(y), dtype = torch::torch_double())

  v <- (y - locations) / scales
  cauchy_factor <- 1 + v[2]^2
  skew_multiplier <- c(1, sqrt(2 / cauchy_factor))
  skew_arg <- alpha * v * skew_multiplier
  skew_cdf <- c(stats::pnorm(skew_arg[1]), stats::pt(skew_arg[2], df = 2))
  skew_score <- c(stats::dnorm(skew_arg[1]), stats::dt(skew_arg[2], df = 2)) /
    skew_cdf
  density <- c(stats::dnorm(v[1]), stats::dcauchy(v[2])) * 2 * skew_cdf / scales
  mixture_density <- sum(weights * density)
  expected <- -log(mixture_density) + lambda_alpha * mean(alpha^2) +
    lambda_nu_inv * mean(nu^(-2))

  # Differentiate the component log densities with respect to v and alpha.
  tau <- weights * density / mixture_density
  score_v <- c(-v[1], -2 * v[2] / cauchy_factor) +
    skew_score * alpha * skew_multiplier / c(1, cauchy_factor)
  expected_gradients <- list(
    pi = -density / mixture_density,
    mu = tau * score_v / scales,
    scale_chol = tau * (1 + v * score_v) / scales,
    alpha = -tau * skew_score * v * skew_multiplier +
      2 * lambda_alpha * alpha / length(alpha)
  )
  loss <- loss_mst_pmdn(pred, target, lambda_alpha, lambda_nu_inv)
  expect_lt(abs(loss$item() - expected), 1e-13)
  loss$backward()
  for (name in names(expected_gradients)) {
    expect_equal(
      normal_test_values(pred[[name]]$grad),
      expected_gradients[[name]],
      tolerance = 1e-13,
      info = name
    )
  }
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
  nu_float <- torch::torch_tensor(nu_values, dtype = torch::torch_float())
  nu_double <- torch::torch_tensor(nu_values, dtype = torch::torch_double())
  for (d in 1:5) {
    actual_float <- MST.PMDN:::.log_multivariate_t_normalizer(nu_float, d)
    actual_double <- MST.PMDN:::.log_multivariate_t_normalizer(nu_double, d)
    expected <- lgamma((nu_values + d) / 2) -
                lgamma(nu_values / 2) -
                (d / 2) * log(nu_values * pi)
    expect_equal(normal_test_values(actual_float), expected, tolerance = 2e-6,
                 info = paste("dimension", d))
    expect_lt(max(abs(normal_test_values(actual_double) - expected)), 5e-14)
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
    alpha = array(0, c(1, 1, 1)),
    skew_none = TRUE
  )

  set.seed(17)
  torch::torch_manual_seed(17)
  draws <- normal_test_values(
    sample_mst_pmdn(pred, num_samples = 20000)$samples
  )

  expect_equal(mean(draws), 0, tolerance = 0.025)
  expect_equal(stats::var(draws), 1, tolerance = 0.04)
})
