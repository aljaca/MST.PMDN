test_that("default path uses plain linear MLP layers and normalized heads", {
  model <- define_mst_pmdn(
    input_dim = 2,
    output_dim = 1,
    hidden_dim = c(5, 3),
    n_mixtures = 2,
    constraint = "VIINN",
    drop_hidden = 0.1
  )

  expect_true(inherits(model$hidden, "nn_sequential"))
  expect_true(inherits(model$hidden[[1]], "nn_linear"))
  expect_true(inherits(model$hidden[[5]], "nn_linear"))
  expect_true(inherits(model$fc_pi, "weight_norm_linear"))
  expect_true(inherits(model$fc_mu, "weight_norm_linear"))
})

test_that("custom fusion bypasses the default MLP", {
  fusion_with_dim <- torch::nn_module(
    "fusion_with_dim",
    initialize = function(input_dim = 2, output_dim = 4) {
      self$output_dim <- output_dim
      self$projection <- torch::nn_linear(input_dim, output_dim)
    },
    forward = function(x) {
      self$projection(x)
    }
  )
  fusion <- fusion_with_dim()

  model <- define_mst_pmdn(
    input_dim = 2,
    output_dim = 1,
    hidden_dim = integer(0),
    n_mixtures = 2,
    constraint = "VIINN",
    drop_hidden = 0.1,
    fusion_module = fusion
  )

  expect_null(model$hidden)
  expect_true(inherits(model$fusion_dropout, "nn_dropout"))
  expect_equal(model$final_hidden_dim, 4)
  pred <- model(torch::torch_randn(c(3, 2)))
  expect_equal(as.integer(pred$mu$size()), c(3L, 2L, 1L))
})

test_that("custom fusion can use the final hidden_dim as a width fallback", {
  fusion_without_dim <- torch::nn_module(
    "fusion_without_dim",
    initialize = function(input_dim = 2, output_dim = 5) {
      self$projection <- torch::nn_linear(input_dim, output_dim)
    },
    forward = function(x) {
      self$projection(x)
    }
  )
  fusion <- fusion_without_dim()

  expect_warning(
    model <- define_mst_pmdn(
      input_dim = 2,
      output_dim = 1,
      hidden_dim = c(7, 5),
      n_mixtures = 2,
      constraint = "VIINN",
      fusion_module = fusion
    ),
    "Using fallback dimension: 5"
  )

  expect_null(model$hidden)
  expect_equal(model$final_hidden_dim, 5)
  pred <- model(torch::torch_randn(c(3, 2)))
  expect_equal(as.integer(pred$mu$size()), c(3L, 2L, 1L))
})
