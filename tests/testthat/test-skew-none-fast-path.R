make_zero_skew_output <- function(B = 2L, M = 2L, d = 3L,
                                  nu_values = 8, skew_none = NULL) {
  weights <- seq_len(M)
  weights <- weights / sum(weights)
  pi <- matrix(rep(weights, B), nrow = B, byrow = TRUE)
  mu <- array(
    seq(-0.2, 0.2, length.out = B * M * d),
    dim = c(B, M, d)
  )
  scale_chol <- array(0, dim = c(B, M, d, d))
  for (b in seq_len(B)) {
    for (m in seq_len(M)) {
      scale_chol[b, m, , ] <- diag(
        0.8 + 0.05 * seq_len(d),
        nrow = d,
        ncol = d
      )
    }
  }
  nu <- matrix(
    rep(nu_values, length.out = B * M),
    nrow = B,
    byrow = TRUE
  )
  make_mdn_output(
    pi = pi,
    mu = mu,
    scale_chol = scale_chol,
    nu = nu,
    alpha = array(0, dim = c(B, M, d)),
    skew_none = skew_none
  )
}

test_that("forward and prediction outputs propagate the structural flag", {
  model <- define_mst_pmdn(
    input_dim = 2,
    output_dim = 1,
    hidden_dim = c(3),
    n_mixtures = 2,
    constraint = "VIINN"
  )
  pred <- predict_mst_pmdn(model, matrix(0, nrow = 3, ncol = 2))
  expect_identical(pred$skew_none, TRUE)

  # A symmetric sampler must not transfer or gather alpha.
  pred$alpha <- structure("unused alpha sentinel", class = "unused_alpha")
  expect_no_error(sample_mst_pmdn(pred, num_samples = 2))

  seen <- logical()
  original_sampler <- MST.PMDN::sample_mst_pmdn
  testthat::local_mocked_bindings(
    sample_mst_pmdn = function(mdn_output, ...) {
      seen <<- c(seen, mdn_output$skew_none)
      original_sampler(mdn_output, ...)
    },
    .package = "MST.PMDN"
  )

  expect_no_error(sample_mst_pmdn_df(pred, num_samples = 2))
  expect_no_error(cdf_marginal_mst_pmdn(
    pred, y = 0, num_samples = 3
  ))
  expect_no_error(quantile_marginal_mst_pmdn(
    pred, probs = matrix(0.5, nrow = 3), num_samples = 3
  ))
  expect_identical(seen, rep(TRUE, 3))
})

test_that("skew_none validation accepts absence and rejects malformed values", {
  expect_identical(
    MST.PMDN:::.validate_skew_none(list()),
    FALSE
  )
  expect_identical(
    MST.PMDN:::.validate_skew_none(list(skew_none = TRUE)),
    TRUE
  )
  expect_identical(
    MST.PMDN:::.validate_skew_none(list(skew_none = FALSE)),
    FALSE
  )

  pred <- make_zero_skew_output(skew_none = TRUE)
  target <- torch::torch_zeros(c(2, 3))
  malformed <- list(
    NA,
    0,
    1,
    "TRUE",
    c(TRUE, FALSE)
  )
  for (value in malformed) {
    loss_output <- pred
    loss_output$skew_none <- value
    expect_error(
      loss_mst_pmdn(loss_output, target),
      "output\\$skew_none must be a single non-missing logical value\\."
    )

    sample_output <- pred
    sample_output$skew_none <- value
    expect_error(
      sample_mst_pmdn(sample_output, num_samples = 1),
      "mdn_output\\$skew_none must be a single non-missing logical value\\."
    )
  }
})

test_that("symmetric loss matches the zero-alpha general formulation", {
  cases <- list(
    list(B = 3L, M = 1L, d = 1L, nu = 7),
    list(B = 2L, M = 3L, d = 3L, nu = c(4, 11, 30)),
    list(B = 2L, M = 2L, d = 5L, nu = c(Inf, 9)),
    list(B = 3L, M = 4L, d = 2L, nu = c(Inf, 5, 15, Inf))
  )

  for (case in cases) {
    fast <- make_zero_skew_output(
      B = case$B, M = case$M, d = case$d,
      nu_values = case$nu, skew_none = TRUE
    )
    general <- make_zero_skew_output(
      B = case$B, M = case$M, d = case$d,
      nu_values = case$nu, skew_none = FALSE
    )
    target <- torch::torch_tensor(
      matrix(
        seq(-0.4, 0.6, length.out = case$B * case$d),
        nrow = case$B
      ),
      dtype = torch::torch_float()
    )

    expect_equal(
      loss_mst_pmdn(fast, target)$item(),
      loss_mst_pmdn(general, target)$item(),
      tolerance = 2e-6,
      info = sprintf("B=%d, M=%d, d=%d", case$B, case$M, case$d)
    )
  }
})

make_loss_gradient_case <- function(skew_none) {
  B <- 3L
  M <- 2L
  d <- 3L
  logits <- torch::torch_tensor(
    matrix(c(0.2, -0.1, -0.3, 0.4, 0.1, -0.2), nrow = B, byrow = TRUE),
    dtype = torch::torch_float(),
    requires_grad = TRUE
  )
  mu <- torch::torch_tensor(
    array(seq(-0.25, 0.3, length.out = B * M * d), c(B, M, d)),
    dtype = torch::torch_float(),
    requires_grad = TRUE
  )
  scale_chol_values <- array(0, c(B, M, d, d))
  for (b in seq_len(B)) {
    for (m in seq_len(M)) {
      scale_chol_values[b, m, , ] <- matrix(
        c(
          0.9, 0, 0,
          0.1, 1.1, 0,
          -0.05, 0.15, 0.8
        ),
        nrow = d,
        byrow = TRUE
      )
    }
  }
  scale_chol <- torch::torch_tensor(
    scale_chol_values,
    dtype = torch::torch_float(),
    requires_grad = TRUE
  )
  nu <- torch::torch_tensor(
    matrix(c(6, 14, 7, 16, 8, 18), nrow = B, byrow = TRUE),
    dtype = torch::torch_float(),
    requires_grad = TRUE
  )
  output <- list(
    pi = torch::nnf_softmax(logits, dim = 2),
    mu = mu,
    scale_chol = scale_chol,
    nu = nu,
    alpha = torch::torch_zeros(c(B, M, d)),
    skew_none = skew_none
  )
  list(
    output = output,
    leaves = list(
      mixture_logits = logits,
      locations = mu,
      scale_chol = scale_chol,
      nu = nu
    )
  )
}

test_that("symmetric bypass preserves all non-alpha gradients", {
  fast <- make_loss_gradient_case(TRUE)
  general <- make_loss_gradient_case(FALSE)
  target <- torch::torch_tensor(
    matrix(seq(-0.5, 0.7, length.out = 9), nrow = 3),
    dtype = torch::torch_float()
  )

  fast_loss <- loss_mst_pmdn(fast$output, target)
  general_loss <- loss_mst_pmdn(general$output, target)
  expect_equal(fast_loss$item(), general_loss$item(), tolerance = 2e-6)
  fast_loss$backward()
  general_loss$backward()

  for (name in names(fast$leaves)) {
    fast_grad <- torch::as_array(fast$leaves[[name]]$grad)
    general_grad <- torch::as_array(general$leaves[[name]]$grad)
    expect_true(all(is.finite(fast_grad)), info = name)
    expect_true(all(is.finite(general_grad)), info = name)
    expect_equal(fast_grad, general_grad, tolerance = 2e-5, info = name)
  }
})

test_that("loss bypass control flow follows only a valid true flag", {
  pred <- make_zero_skew_output(B = 2, M = 2, d = 3, skew_none = TRUE)
  target <- torch::torch_zeros(c(2, 3))
  calls <- 0L
  testthat::local_mocked_bindings(
    .log_skew_factor_mst = function(alpha, v, nu_safe, maha, normal, d) {
      calls <<- calls + 1L
      torch::torch_zeros_like(maha)
    },
    .package = "MST.PMDN"
  )

  loss_mst_pmdn(pred, target)
  expect_identical(calls, 0L)

  pred$skew_none <- FALSE
  loss_mst_pmdn(pred, target)
  expect_identical(calls, 1L)

  pred$skew_none <- NULL
  loss_mst_pmdn(pred, target)
  expect_identical(calls, 2L)
})

test_that("inactive penalties allocate no tensors or alpha reduction graph", {
  pred <- make_zero_skew_output(
    B = 2, M = 2, d = 3, nu_values = 8, skew_none = TRUE
  )
  pred$alpha <- structure("unused alpha sentinel", class = "unused_alpha")
  target <- torch::torch_zeros(c(2, 3))
  tensor_calls <- 0L
  original_tensor <- torch::torch_tensor
  testthat::local_mocked_bindings(
    torch_tensor = function(...) {
      tensor_calls <<- tensor_calls + 1L
      original_tensor(...)
    },
    .package = "MST.PMDN"
  )

  expect_no_error(loss_mst_pmdn(pred, target))
  expect_identical(tensor_calls, 0L)

  expect_no_error(loss_mst_pmdn(pred, target, lambda_alpha = 0.5))
  expect_identical(tensor_calls, 0L)

  expect_no_error(loss_mst_pmdn(pred, target, lambda_nu_inv = 0.5))
  expect_identical(tensor_calls, 1L)
})

test_that("loss validates penalty weights and requires alpha", {
  pred <- make_zero_skew_output(
    B = 2, M = 2, d = 3, nu_values = 8, skew_none = TRUE
  )
  target <- torch::torch_zeros(c(2, 3))
  invalid <- list(
    NA_real_,
    NaN,
    Inf,
    -Inf,
    -0.1,
    numeric(),
    c(0, 1),
    "0",
    FALSE,
    torch::torch_tensor(0)
  )

  for (value in invalid) {
    expect_error(
      loss_mst_pmdn(pred, target, lambda_alpha = value),
      "lambda_alpha must be a single finite non-negative numeric value\\."
    )
    expect_error(
      loss_mst_pmdn(pred, target, lambda_nu_inv = value),
      "lambda_nu_inv must be a single finite non-negative numeric value\\."
    )
  }

  missing_alpha <- pred
  missing_alpha$alpha <- NULL
  expect_error(
    loss_mst_pmdn(missing_alpha, target),
    "output\\$alpha is required\\."
  )
})

test_that("sampler requires alpha for every skew_none state", {
  for (skew_none in list(TRUE, FALSE, NULL)) {
    pred <- make_zero_skew_output(skew_none = skew_none)
    pred$alpha <- NULL
    expect_error(
      sample_mst_pmdn(pred, num_samples = 1),
      "mdn_output\\$alpha is required\\."
    )
  }
})

test_that("sampler bypass draws one latent tensor and never reads alpha", {
  B <- 2L
  S <- 4L
  d <- 3L
  shapes <- list()
  original_randn <- torch::torch_randn
  testthat::local_mocked_bindings(
    torch_randn = function(...) {
      args <- list(...)
      shapes[[length(shapes) + 1L]] <<- as.integer(args[[1]])
      do.call(original_randn, args)
    },
    .package = "MST.PMDN"
  )

  fast <- make_zero_skew_output(
    B = B, M = 2, d = d, nu_values = Inf, skew_none = TRUE
  )
  fast$alpha <- structure("unused alpha sentinel", class = "unused_alpha")
  expect_no_error(sample_mst_pmdn(fast, num_samples = S))
  expect_equal(shapes, list(c(B, S, d)))

  general <- make_zero_skew_output(
    B = B, M = 2, d = d, nu_values = Inf, skew_none = FALSE
  )
  expect_no_error(sample_mst_pmdn(general, num_samples = S))
  expect_equal(
    shapes[2:3],
    list(c(B, S, 1L), c(B, S, d))
  )

  general$skew_none <- NULL
  expect_no_error(sample_mst_pmdn(general, num_samples = S))
  expect_equal(
    shapes[4:5],
    list(c(B, S, 1L), c(B, S, d))
  )
})
