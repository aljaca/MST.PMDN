make_mdn_output <- function(pi, mu, scale_chol, nu, alpha,
                            skew_none = NULL,
                            dtype = torch::torch_float(),
                            device = "cpu") {
  as_test_tensor <- function(values) {
    # Own the storage even when torch can share the source R array.
    torch::torch_tensor(values, dtype = dtype, device = device)$clone()
  }
  output <- list(
    pi = as_test_tensor(pi),
    mu = as_test_tensor(mu),
    scale_chol = as_test_tensor(scale_chol),
    nu = as_test_tensor(nu),
    alpha = as_test_tensor(alpha)
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
