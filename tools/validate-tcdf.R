#!/usr/bin/env Rscript

options(width = 120)

if (!file.exists(file.path("R", "MST-PMDN.R"))) {
  stop("Run this script from the package root.")
}

suppressPackageStartupMessages(library(torch))
source(file.path("R", "MST-PMDN.R"))

tensor_values <- function(x) {
  as.numeric(torch::as_array(x$to(device = "cpu")))
}

probabilities <- seq(1e-8, 1 - 1e-8, length.out = 10001)
degrees_of_freedom <- c(3, 4, 5, 6, 10, 30, 100, 502)

cdf_results <- do.call(rbind, lapply(degrees_of_freedom, function(df) {
  quantiles <- stats::qt(probabilities, df = df)
  approximate <- t_cdf(
    torch_tensor(quantiles, dtype = torch_double()),
    torch_tensor(df, dtype = torch_double())
  )
  data.frame(
    df = df,
    max_absolute_error = max(abs(tensor_values(approximate) - probabilities)),
    threshold = if (df == 3) 5e-3 else 2e-3
  )
}))

tail_cases <- data.frame(
  df = c(5, 5, 5, 10, 30),
  probability = c(1e-3, 1e-4, 1e-6, 1e-6, 1e-6)
)
tail_quantiles <- stats::qt(tail_cases$probability, df = tail_cases$df)
tail_approximate <- log_pt(
  torch_tensor(tail_quantiles, dtype = torch_double()),
  torch_tensor(tail_cases$df, dtype = torch_double())
)
tail_cases$absolute_log_error <- abs(
  tensor_values(tail_approximate) - log(tail_cases$probability)
)

z_values <- stats::qt(c(1e-3, 1e-4, 1e-6), df = 5)
z <- torch_tensor(z_values, dtype = torch_double(), requires_grad = TRUE)
nu <- torch_tensor(rep(5, length(z_values)),
                   dtype = torch_double(), requires_grad = TRUE)
log_cdf <- log_pt(z, nu)
log_cdf$sum()$backward()

exact_z_gradient <- stats::dt(z_values, df = 5) /
                    stats::pt(z_values, df = 5)
h <- 1e-4
exact_nu_gradient <- (
  stats::pt(z_values, df = 5 + h, log.p = TRUE) -
  stats::pt(z_values, df = 5 - h, log.p = TRUE)
) / (2 * h)
gradient_results <- data.frame(
  probability = c(1e-3, 1e-4, 1e-6),
  z_relative_error = abs(tensor_values(z$grad) / exact_z_gradient - 1),
  nu_relative_error = abs(tensor_values(nu$grad) / exact_nu_gradient - 1)
)

float32_z <- torch_tensor(
  c(-6, -7, -10, -14),
  dtype = torch_float(),
  requires_grad = TRUE
)
float32_log_cdf <- .log_normal_cdf(float32_z)
float32_log_cdf$sum()$backward()
float32_results <- data.frame(
  z = c(-6, -7, -10, -14),
  approximate_log_cdf = tensor_values(float32_log_cdf),
  exact_log_cdf = stats::pnorm(c(-6, -7, -10, -14), log.p = TRUE),
  gradient = tensor_values(float32_z$grad)
)

cat("CDF error by degrees of freedom\n")
print(cdf_results, row.names = FALSE)
cat("\nLower-tail log-CDF error\n")
print(tail_cases, row.names = FALSE)
cat("\nAutograd relative error\n")
print(gradient_results, row.names = FALSE)
cat("\nFloat32 normal log-CDF\n")
print(float32_results, row.names = FALSE)

stopifnot(
  all(cdf_results$max_absolute_error < cdf_results$threshold),
  all(tail_cases$absolute_log_error <
      c(0.05, 0.12, 0.32, 0.10, 0.02)),
  max(gradient_results$z_relative_error) < 0.06,
  max(gradient_results$nu_relative_error) < 0.06,
  all(is.finite(float32_results$approximate_log_cdf)),
  all(is.finite(float32_results$gradient)),
  all(float32_results$gradient > 0),
  max(abs(float32_results$approximate_log_cdf -
          float32_results$exact_log_cdf)) < 5e-4
)

cat("\nAll validation thresholds passed.\n")
