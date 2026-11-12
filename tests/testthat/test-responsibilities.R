make_univariate_responsibility_prediction <- function(
    weights, mu = 0, nu = Inf, alpha = 0, skew_none = TRUE,
    dtype = torch::torch_double(), device = "cpu") {
  B <- nrow(weights)
  M <- ncol(weights)
  make_mdn_output(
    pi = weights,
    mu = array(mu, c(B, M, 1L)),
    scale_chol = array(1, c(B, M, 1L, 1L)),
    nu = matrix(nu, B, M),
    alpha = array(alpha, c(B, M, 1L)),
    skew_none = skew_none,
    dtype = dtype,
    device = device
  )
}

test_that("responsibilities match mixed joint densities and preserve affine changes", {
  weights <- rbind(c(0.5, 0.3, 0.2), c(0.2, 0.5, 0.3), c(0.1, 0.4, 0.5))
  target <- rbind(c(-0.8, 0.4), c(0.2, -0.7), c(1.3, 0.5))
  locations <- rbind(c(0, 0.2), c(-0.5, 0.1), c(0.8, -0.3))
  slants <- rbind(c(0, 0), c(0.8, -0.3), c(-0.4, 0.5))
  nu <- c(Inf, Inf, 8)
  factors <- list(
    matrix(c(1.1, 0.35, 0, 0.8), 2),
    matrix(c(0.9, -0.25, 0, 1.2), 2),
    matrix(c(1.2, 0.2, 0, 0.7), 2)
  )
  scale_chol <- array(0, c(3, 3, 2, 2))
  log_density <- matrix(0, 3, 3)
  for (k in seq_len(3)) {
    L <- factors[[k]]
    for (i in seq_len(3)) {
      scale_chol[i, k, , ] <- L
    }
    Sigma <- tcrossprod(L)
    q <- stats::mahalanobis(target, locations[k, ], Sigma)
    log_det <- as.numeric(determinant(Sigma, logarithm = TRUE)$modulus)
    omega <- sqrt(diag(Sigma))
    alpha_sn <- omega * backsolve(t(L), slants[k, ])
    residuals <- sweep(target, 2, locations[k, ], "-")
    z <- drop(sweep(residuals, 2, omega, "/") %*% alpha_sn)
    if (is.infinite(nu[k])) {
      kernel <- -log(2 * pi) - 0.5 * (log_det + q)
      log_cdf <- stats::pnorm(z, log.p = TRUE)
    } else {
      df <- nu[k] + 2
      kernel <- lgamma(df / 2) - lgamma(nu[k] / 2) - log(nu[k] * pi) -
        0.5 * log_det - (df / 2) * log1p(q / nu[k])
      log_cdf <- stats::pt(z * sqrt(df / (nu[k] + q)), df = df, log.p = TRUE)
    }
    log_density[, k] <- kernel + log(2) + log_cdf
  }
  pred <- make_mdn_output(
    pi = weights,
    mu = array(rep(as.vector(locations), each = 3), c(3, 3, 2)),
    scale_chol = scale_chol,
    nu = matrix(rep(nu, each = 3), 3, 3),
    alpha = array(rep(as.vector(slants), each = 3), c(3, 3, 2)),
    dtype = torch::torch_double()
  )
  expected <- weights * exp(log_density)
  expected <- expected / rowSums(expected)
  actual <- responsibilities_mst_pmdn(pred, target, as_array = TRUE)
  expect_lt(max(abs(actual - expected)), 2e-6)
  expect_equal(rowSums(actual), rep(1, 3), tolerance = 1e-14)
  expect_equal(
    exp(responsibilities_mst_pmdn(pred, target, log = TRUE, as_array = TRUE)),
    actual,
    tolerance = 1e-14
  )

  float_pred <- lapply(pred, function(value) value$to(dtype = torch::torch_float()))
  float_result <- responsibilities_mst_pmdn(float_pred, target)
  expect_identical(as.character(float_result$dtype), "Float")
  expect_lt(max(abs(torch::as_array(float_result) - actual)), 1e-6)

  stretch <- c(0.75, 2.25)
  offset <- c(-0.3, 1.2)
  scale <- torch::torch_tensor(stretch, dtype = torch::torch_double())
  shift <- torch::torch_tensor(offset, dtype = torch::torch_double())
  transformed <- pred
  transformed$mu <- pred$mu * scale + shift
  transformed$scale_chol <- pred$scale_chol * scale$view(c(1, 1, 2, 1))
  new_target <- sweep(sweep(target, 2, stretch, "*"), 2, offset, "+")
  expect_equal(
    responsibilities_mst_pmdn(transformed, new_target, as_array = TRUE),
    actual,
    tolerance = 1e-12
  )
})

test_that("skew-normal responsibilities follow both normal log-CDF branches", {
  target <- matrix(c(-9.9, -10, -10.001, -13), ncol = 1)
  pred <- make_univariate_responsibility_prediction(
    matrix(0.5, 4, 2), alpha = rep(c(0, 1), each = 4), skew_none = FALSE
  )
  log_ratio <- log(2) + stats::pnorm(target[, 1], log.p = TRUE)
  expected <- log_ratio - log1p(exp(log_ratio))
  actual <- responsibilities_mst_pmdn(
    pred, target, log = TRUE, as_array = TRUE
  )[, 2]
  error <- abs(actual - expected)
  expect_lt(max(error[target[, 1] >= -10]), 1e-12)
  expect_lt(max(error[target[, 1] < -10]), 1e-6)
})

test_that("zero weights are inactive with finite entropy gradients", {
  weights <- matrix(c(0, 0.3, 0.7, 1e-20), nrow = 1)
  pred <- make_univariate_responsibility_prediction(weights)
  pred$pi$requires_grad_(TRUE)
  target <- matrix(0, 1, 1)
  probabilities <- responsibilities_mst_pmdn(pred, target, as_array = TRUE)
  expect_identical(probabilities[1, 1], 0)
  expect_equal(probabilities[1, 4] / weights[1, 4], 1, tolerance = 1e-14)

  log_probs <- responsibilities_mst_pmdn(pred, target, log = TRUE)
  expect_identical(log_probs[1, 1]$item(), -Inf)
  expect_true(log_probs$requires_grad)
  safe_logs <- log_probs$masked_fill(!torch::torch_isfinite(log_probs), 0)
  entropy <- -(log_probs$exp() * safe_logs)$sum()
  entropy$backward()
  gradient <- torch::as_array(pred$pi$grad)
  expect_true(all(is.finite(gradient)))
  expect_identical(gradient[1, 1], 0)
  expect_gt(abs(gradient[1, 2]), 0)
})

test_that("log responsibilities stay finite after probabilities underflow", {
  pred <- make_univariate_responsibility_prediction(
    matrix(c(0.5, 0.5), nrow = 1), mu = c(0, 20), dtype = torch::torch_float()
  )
  target <- matrix(0, 1, 1)
  expect_identical(responsibilities_mst_pmdn(pred, target)[1, 2]$item(), 0)
  expect_equal(
    responsibilities_mst_pmdn(pred, target, log = TRUE)[1, 2]$item(),
    -200,
    tolerance = 1e-7
  )
})

test_that("responsibilities use the density cap and preserve float32 weights", {
  weights <- matrix(c(0.55, 0.45), nrow = 1)
  two_capped <- make_univariate_responsibility_prediction(
    weights, mu = c(1500, 2000), dtype = torch::torch_float()
  )
  target <- matrix(0, 1, 1)
  expect_equal(
    responsibilities_mst_pmdn(two_capped, target, as_array = TRUE),
    weights,
    tolerance = 1e-7
  )

  one_capped <- make_univariate_responsibility_prediction(
    matrix(c(0.9999, 0.0001), nrow = 1), mu = c(2000, 0), nu = c(1, Inf)
  )
  cauchy <- 0.9999 * stats::dcauchy(1000)
  gaussian <- 0.0001 * stats::dnorm(0)
  expected <- cauchy / (cauchy + gaussian)
  actual <- responsibilities_mst_pmdn(one_capped, target, as_array = TRUE)[1, 1]
  expect_lt(abs(actual - expected), 1e-16)
})

test_that("responsibilities validate target alignment and posterior rows", {
  pred <- make_univariate_responsibility_prediction(matrix(1, 2, 1))
  attr(pred, "response_names") <- "flow"
  target <- matrix(c(0, 0.5), ncol = 1, dimnames = list(NULL, "flow"))
  expect_identical(
    responsibilities_mst_pmdn(pred, target, as_array = TRUE),
    matrix(1, 2, 1)
  )
  expect_error(responsibilities_mst_pmdn(pred, t(target)), "one row per prediction")
  expect_error(responsibilities_mst_pmdn(pred, 0), "real numeric matrix")
  expect_error(responsibilities_mst_pmdn(pred, target + NA_real_), "finite responses")
  expect_error(responsibilities_mst_pmdn(pred, target, log = NA), "log must be")
  expect_error(responsibilities_mst_pmdn(pred, target, as_array = 1), "as_array must be")
  colnames(target) <- "wrong"
  expect_error(responsibilities_mst_pmdn(pred, target), "response_names in order")
  colnames(target) <- NULL

  for (weight in c(0, -1, Inf, NaN)) {
    invalid <- make_univariate_responsibility_prediction(matrix(weight, 2, 1))
    expect_error(responsibilities_mst_pmdn(invalid, target), "non-negative weights")
  }
  invalid <- make_univariate_responsibility_prediction(matrix(1, 2, 1), mu = NaN)
  expect_error(responsibilities_mst_pmdn(invalid, target), "finite maximum")
})

test_that("predictions and tracked forward passes give the same responsibilities", {
  torch::torch_manual_seed(524L)
  model <- define_mst_pmdn(
    input_dim = 1, output_dim = 1, hidden_dim = 3, n_mixtures = 2,
    constraint = "VIINN"
  )
  inputs <- matrix(c(-0.5, 0.7, 1.1), ncol = 1)
  target <- torch::torch_tensor(matrix(c(0.3, -0.6, 0.2), ncol = 1))
  pred <- predict_mst_pmdn(model, inputs)
  probabilities <- responsibilities_mst_pmdn(pred, target, log = TRUE)
  expect_false(probabilities$requires_grad)

  model$eval()
  tracked_pred <- model(torch::torch_tensor(inputs, dtype = pred$mu$dtype))
  tracked <- responsibilities_mst_pmdn(tracked_pred, target, log = TRUE)
  expect_true(tracked$requires_grad)
  expect_equal(
    torch::as_array(tracked$detach()),
    torch::as_array(probabilities),
    tolerance = 1e-7
  )
  tracked[, 1]$sum()$backward()
  gradient <- torch::as_array(model$fc_pi$bias$grad)
  expect_true(all(is.finite(gradient)))
  expect_gt(sum(abs(gradient)), 0)
})

test_that("responsibilities follow the prediction device", {
  skip_if_not(torch::cuda_is_available(), "CUDA is not available")
  pred <- make_univariate_responsibility_prediction(
    matrix(c(0.25, 0.75), nrow = 1), device = "cuda"
  )
  target <- torch::torch_tensor(matrix(0, 1, 1))
  result <- responsibilities_mst_pmdn(pred, target)
  expect_true(result$is_cuda)
  expect_identical(as.character(result$dtype), "Double")
  expect_equal(
    responsibilities_mst_pmdn(pred, target, as_array = TRUE),
    matrix(c(0.25, 0.75), nrow = 1),
    tolerance = 1e-12
  )
})
