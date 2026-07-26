tensor_values <- function(x) {
  as.numeric(torch::as_array(x$to(device = "cpu")))
}

test_that("Gaver-Kafadar CDF has bounded absolute error", {
  probs <- seq(1e-8, 1 - 1e-8, length.out = 10001)
  for (df in c(3, 4, 5, 6, 10, 30, 100, 502)) {
    z <- stats::qt(probs, df = df)
    approx <- t_cdf(
      torch::torch_tensor(z, dtype = torch::torch_double()),
      torch::torch_tensor(rep(df, length(z)), dtype = torch::torch_double())
    )
    threshold <- if (df == 3) 5e-3 else 2e-3
    expect_lt(max(abs(tensor_values(approx) - probs)), threshold)
  }
})

test_that("Gaver-Kafadar log-CDF is accurate in the lower tail", {
  cases <- data.frame(
    df = c(5, 5, 5, 10, 30),
    p = c(1e-3, 1e-4, 1e-6, 1e-6, 1e-6),
    tolerance = c(0.05, 0.12, 0.32, 0.10, 0.02)
  )
  z <- stats::qt(cases$p, df = cases$df)
  approx <- MST.PMDN:::log_pt(
    torch::torch_tensor(z, dtype = torch::torch_double()),
    torch::torch_tensor(cases$df, dtype = torch::torch_double())
  )
  error <- abs(tensor_values(approx) - log(cases$p))
  expect_true(all(error < cases$tolerance))
})

test_that("t CDF is symmetric, monotone, and bounded", {
  z <- seq(-20, 20, length.out = 4001)
  for (df in c(4, 5, 10, 30, 502)) {
    approx <- tensor_values(t_cdf(
      torch::torch_tensor(z, dtype = torch::torch_double()),
      torch::torch_tensor(df, dtype = torch::torch_double())
    ))
    expect_true(all(is.finite(approx)))
    expect_true(all(approx >= 0 & approx <= 1))
    expect_true(all(diff(approx) >= -1e-12))
    expect_equal(approx + rev(approx), rep(1, length(approx)),
                 tolerance = 1e-11)
  }
})

test_that("exact CDF branches for one and two degrees of freedom are stable", {
  z <- c(-1e6, -100, -5, -1, 0, 1, 5, 100, 1e6)
  for (df in c(1, 2)) {
    approx <- tensor_values(t_cdf(
      torch::torch_tensor(z, dtype = torch::torch_double()),
      torch::torch_tensor(df, dtype = torch::torch_double())
    ))
    expect_equal(approx, stats::pt(z, df = df), tolerance = 1e-10)
  }
})

test_that("stable normal log-CDF remains finite in the float32 tail", {
  z <- torch::torch_tensor(
    c(-6, -7, -10, -14),
    dtype = torch::torch_float(),
    requires_grad = TRUE
  )
  log_cdf <- MST.PMDN:::.log_normal_cdf(z)
  expect_true(all(is.finite(tensor_values(log_cdf))))
  expect_equal(
    tensor_values(log_cdf),
    stats::pnorm(c(-6, -7, -10, -14), log.p = TRUE),
    tolerance = 5e-4
  )
  log_cdf$sum()$backward()
  gradient <- tensor_values(z$grad)
  expect_true(all(is.finite(gradient)))
  expect_true(all(gradient > 0))
})

test_that("Student t log-CDF has no lower-tail probability floor", {
  z <- torch::torch_tensor(
    -1e6,
    dtype = torch::torch_float(),
    requires_grad = TRUE
  )
  log_cdf <- MST.PMDN:::log_pt(z, 5)
  expect_true(is.finite(log_cdf$item()))
  expect_lt(log_cdf$item(), log(1e-12))
  log_cdf$backward()
  expect_true(is.finite(z$grad$item()))
  expect_gt(z$grad$item(), 0)
})

test_that("log-CDF gradients agree with exact reference derivatives", {
  probs <- c(1e-3, 1e-4, 1e-6)
  df <- 5
  z_values <- stats::qt(probs, df = df)
  z <- torch::torch_tensor(
    z_values,
    dtype = torch::torch_double(),
    requires_grad = TRUE
  )
  nu <- torch::torch_tensor(
    rep(df, length(probs)),
    dtype = torch::torch_double(),
    requires_grad = TRUE
  )
  log_cdf <- MST.PMDN:::log_pt(z, nu)
  log_cdf$sum()$backward()

  exact_z_gradient <- stats::dt(z_values, df = df) /
                      stats::pt(z_values, df = df)
  h <- 1e-4
  exact_nu_gradient <- (
    stats::pt(z_values, df = df + h, log.p = TRUE) -
    stats::pt(z_values, df = df - h, log.p = TRUE)
  ) / (2 * h)

  z_relative_error <- abs(tensor_values(z$grad) / exact_z_gradient - 1)
  nu_relative_error <- abs(tensor_values(nu$grad) / exact_nu_gradient - 1)
  expect_lt(max(z_relative_error), 0.06)
  expect_lt(max(nu_relative_error), 0.06)
})

test_that("log-CDF gradients remain finite at the removable singularity", {
  z <- torch::torch_tensor(
    c(-1e-8, 0, 1e-8),
    dtype = torch::torch_double(),
    requires_grad = TRUE
  )
  nu <- torch::torch_tensor(
    rep(5, 3),
    dtype = torch::torch_double(),
    requires_grad = TRUE
  )
  MST.PMDN:::log_pt(z, nu)$sum()$backward()
  expect_true(all(is.finite(tensor_values(z$grad))))
  expect_true(all(is.finite(tensor_values(nu$grad))))
  expect_true(all(tensor_values(z$grad) > 0))
})

test_that("CPU and CUDA t CDF calculations agree", {
  testthat::skip_if(!torch::cuda_is_available(), "CUDA is not available")
  z_cpu <- torch::torch_tensor(
    c(-10, -5, -1, 0, 1, 5, 10),
    dtype = torch::torch_float()
  )
  nu_cpu <- torch::torch_tensor(
    c(4, 5, 6, 10, 30, 100, 502),
    dtype = torch::torch_float()
  )
  cpu <- tensor_values(t_cdf(z_cpu, nu_cpu))
  cuda <- tensor_values(t_cdf(
    z_cpu$to(device = "cuda"),
    nu_cpu$to(device = "cuda")
  ))
  expect_equal(cuda, cpu, tolerance = 1e-6)
})
