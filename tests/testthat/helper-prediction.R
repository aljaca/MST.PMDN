make_mdn_output <- function(pi, mu, scale_chol, nu, alpha,
                            skew_none = NULL,
                            dtype = torch::torch_float(),
                            device = "cpu") {
  output <- list(
    pi = torch::torch_tensor(pi, dtype = dtype, device = device),
    mu = torch::torch_tensor(mu, dtype = dtype, device = device),
    scale_chol = torch::torch_tensor(
      scale_chol, dtype = dtype, device = device
    ),
    nu = torch::torch_tensor(nu, dtype = dtype, device = device),
    alpha = torch::torch_tensor(alpha, dtype = dtype, device = device)
  )
  if (!is.null(skew_none)) {
    output$skew_none <- skew_none
  }
  output
}

expect_state_dict_equal <- function(x, y, tolerance = 1e-6) {
  expect_setequal(names(x), names(y))
  for (name in names(x)) {
    expect_equal(
      torch::as_array(x[[name]]$cpu()),
      torch::as_array(y[[name]]$cpu()),
      tolerance = tolerance,
      info = name
    )
  }
}
