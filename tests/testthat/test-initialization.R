test_that("specialized nu and alpha initialization survives generic initialization", {
  torch::torch_manual_seed(7)
  model <- define_mst_pmdn(
    input_dim = 2,
    output_dim = 2,
    hidden_dim = c(4),
    n_mixtures = 2,
    constraint = "VIIVV",
    range_nu = c(3, 103)
  )
  model$apply(MST.PMDN:::init_weight_norm)
  MST.PMDN:::init_distribution_heads(model)
  x <- torch::torch_tensor(
    matrix(c(-2, 1, 4, -3), nrow = 2, byrow = TRUE),
    dtype = torch::torch_float()
  )
  pred <- model(x)
  nu <- torch::as_array(pred$nu)
  alpha <- torch::as_array(pred$alpha)

  expect_equal(nu, matrix(53, nrow = 2, ncol = 2), tolerance = 1e-6)
  expect_equal(alpha, array(0, dim = c(2, 2, 2)), tolerance = 1e-6)
})
