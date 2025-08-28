library(torch)
source("R/MST-PMDN.R")

B <- 1; M <- 1; d <- 1

target <- torch_zeros(B, d)
scale_chol <- torch_eye(d)$unsqueeze(1)$unsqueeze(1)$expand(c(B, M, d, d))
output <- list(
  pi = torch_ones(B, M),
  mu = torch_zeros(B, M, d),
  scale_chol = scale_chol,
  nu = torch_full(c(B, M), 5),
  alpha = torch_full(c(B, M, d), 2)
)

loss0 <- loss_mst_pmdn(output, target, lambda_alpha = 0)
loss1 <- loss_mst_pmdn(output, target, lambda_alpha = 0.5)
penalty <- 0.5 * output$alpha$pow(2)$mean()

stopifnot(abs((loss1 - loss0 - penalty)$item()) < 1e-6)
cat("alpha penalty test passed\n")
