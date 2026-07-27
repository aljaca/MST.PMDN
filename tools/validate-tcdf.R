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
    threshold = if (df == 3) 1.2e-5 else 2e-6
  )
}))

tail_cases <- data.frame(
  df = c(5, 5, 5, 10, 30),
  probability = c(1e-4, 1e-6, 1e-9, 1e-9, 1e-9),
  threshold = c(5e-4, 3e-3, 0.03, 3e-4, 1e-5)
)
tail_quantiles <- stats::qt(tail_cases$probability, df = tail_cases$df)
tail_approximate <- log_pt(
  torch_tensor(tail_quantiles, dtype = torch_double()),
  torch_tensor(tail_cases$df, dtype = torch_double())
)
tail_cases$absolute_log_error <- abs(
  tensor_values(tail_approximate) - log(tail_cases$probability)
)

reachable_cases <- expand.grid(
  d = c(1, 2),
  nu = c(3, 5, 10, 30, 100, 500),
  fraction = c(0.25, 0.5, 0.75, 1)
)
reachable_cases$df <- reachable_cases$nu + reachable_cases$d
reachable_cases$z <- -2.5 * sqrt(
  reachable_cases$d * reachable_cases$df
) * reachable_cases$fraction
reachable_approximate <- log_pt(
  torch_tensor(reachable_cases$z, dtype = torch_double()),
  torch_tensor(reachable_cases$df, dtype = torch_double())
)
reachable_cases$absolute_log_error <- abs(
  tensor_values(reachable_approximate) -
  stats::pt(reachable_cases$z, df = reachable_cases$df, log.p = TRUE)
)

gradient_probabilities <- c(1e-4, 1e-6, 1e-9)
gradient_df <- 5.25
z_values <- stats::qt(gradient_probabilities, df = gradient_df)
z <- torch_tensor(z_values, dtype = torch_double(), requires_grad = TRUE)
nu <- torch_tensor(rep(gradient_df, length(z_values)),
                   dtype = torch_double(), requires_grad = TRUE)
log_cdf <- log_pt(z, nu)
log_cdf$sum()$backward()

exact_z_gradient <- stats::dt(z_values, df = gradient_df) /
                    stats::pt(z_values, df = gradient_df)
h <- 1e-4
exact_nu_gradient <- (
  stats::pt(z_values, df = gradient_df + h, log.p = TRUE) -
  stats::pt(z_values, df = gradient_df - h, log.p = TRUE)
) / (2 * h)
gradient_results <- data.frame(
  probability = gradient_probabilities,
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
cat("\nDefault reachable-domain maximum log-CDF error\n")
print(max(reachable_cases$absolute_log_error))
cat("\nAutograd relative error\n")
print(gradient_results, row.names = FALSE)
cat("\nFloat32 normal log-CDF\n")
print(float32_results, row.names = FALSE)

stopifnot(
  all(cdf_results$max_absolute_error < cdf_results$threshold),
  all(tail_cases$absolute_log_error < tail_cases$threshold),
  max(reachable_cases$absolute_log_error) < 3e-4,
  max(gradient_results$z_relative_error) < 0.01,
  max(gradient_results$nu_relative_error) < 0.01,
  all(is.finite(float32_results$approximate_log_cdf)),
  all(is.finite(float32_results$gradient)),
  all(float32_results$gradient > 0),
  max(abs(float32_results$approximate_log_cdf -
          float32_results$exact_log_cdf)) < 5e-4
)

cat("\nAll validation thresholds passed.\n")
