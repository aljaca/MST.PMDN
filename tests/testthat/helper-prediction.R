make_mdn_output <- function(pi, mu, scale_chol, nu, alpha,
                            skew_none = NULL) {
  output <- list(
    pi = torch::torch_tensor(pi, dtype = torch::torch_float()),
    mu = torch::torch_tensor(mu, dtype = torch::torch_float()),
    scale_chol = torch::torch_tensor(
      scale_chol, dtype = torch::torch_float()
    ),
    nu = torch::torch_tensor(nu, dtype = torch::torch_float()),
    alpha = torch::torch_tensor(alpha, dtype = torch::torch_float())
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
