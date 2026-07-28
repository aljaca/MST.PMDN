make_small_model <- function() {
  define_mst_pmdn(
    input_dim = 2,
    output_dim = 1,
    hidden_dim = c(4),
    n_mixtures = 1,
    constraint = "VIINN"
  )
}

test_that("validation uses every case and is batch-size invariant", {
  x <- cbind(seq(-1, 1, length.out = 13), seq(1, -1, length.out = 13))
  y <- matrix(0.5 * x[, 1] - 0.25 * x[, 2], ncol = 1)
  split <- list(train = 1:7, validation = 8:13)

  torch::torch_manual_seed(11)
  base_model <- make_small_model()
  model_a <- make_small_model()
  model_b <- make_small_model()
  model_a$load_state_dict(base_model$state_dict())
  model_b$load_state_dict(base_model$state_dict())

  path_a <- tempfile(fileext = ".pt")
  path_b <- tempfile(fileext = ".pt")
  fit_a <- train_mst_pmdn(
    x, y,
    hidden_dim = c(4), n_mixtures = 1, constraint = "VIINN",
    epochs = 1, lr = 0, batch_size = 4,
    checkpoint_interval = 1, checkpoint_path = path_a,
    model = model_a, custom_split = split,
    scheduler_step = NULL, min_last_batch_frac = 0
  )
  fit_b <- train_mst_pmdn(
    x, y,
    hidden_dim = c(4), n_mixtures = 1, constraint = "VIINN",
    epochs = 1, lr = 0, batch_size = 5,
    checkpoint_interval = 1, checkpoint_path = path_b,
    model = model_b, custom_split = split,
    scheduler_step = NULL, min_last_batch_frac = 0
  )

  expect_equal(fit_a$val_loss_history, fit_b$val_loss_history,
               tolerance = 1e-6)
  pred <- predict_mst_pmdn(fit_a$model, x[split$validation, , drop = FALSE])
  expect_identical(pred$skew_none, TRUE)
  manual_loss <- loss_mst_pmdn(
    pred,
    torch::torch_tensor(y[split$validation, , drop = FALSE],
                        dtype = torch::torch_float())
  )$item()
  expect_equal(fit_a$val_loss_history[[1]], manual_loss, tolerance = 1e-6)
  expect_equal(length(fit_a$val_indices), length(split$validation))
})

test_that("checkpoint resume reproduces uninterrupted training", {
  skip_on_cran()
  x <- cbind(seq(-2, 2, length.out = 16), cos(seq_len(16)))
  y <- matrix(sin(x[, 1]) + 0.2 * x[, 2], ncol = 1)
  split <- list(train = 1:12, validation = 13:16)
  full_path <- tempfile(fileext = ".pt")
  resumed_path <- tempfile(fileext = ".pt")

  common <- list(
    inputs = x,
    outputs = y,
    hidden_dim = c(4),
    n_mixtures = 1,
    constraint = "VIINN",
    lr = 0.005,
    batch_size = 4,
    checkpoint_interval = 2,
    early_stopping_patience = 100,
    scheduler_step = 2,
    scheduler_gamma = 0.8,
    min_last_batch_frac = 0
  )

  set.seed(101)
  torch::torch_manual_seed(101)
  fit_full <- do.call(
    train_mst_pmdn,
    c(common, list(
      epochs = 4,
      checkpoint_path = full_path,
      custom_split = split
    ))
  )

  set.seed(101)
  torch::torch_manual_seed(101)
  do.call(
    train_mst_pmdn,
    c(common, list(
      epochs = 2,
      checkpoint_path = resumed_path,
      custom_split = split
    ))
  )
  fit_resumed <- do.call(
    train_mst_pmdn,
    c(common, list(
      epochs = 4,
      checkpoint_path = resumed_path,
      resume_from_checkpoint = TRUE
    ))
  )

  expect_equal(fit_resumed$train_indices, fit_full$train_indices)
  expect_equal(fit_resumed$val_indices, fit_full$val_indices)
  expect_equal(fit_resumed$train_loss_history,
               fit_full$train_loss_history, tolerance = 1e-6)
  expect_equal(fit_resumed$val_loss_history,
               fit_full$val_loss_history, tolerance = 1e-6)
  expect_state_dict_equal(
    fit_resumed$model$state_dict(),
    fit_full$model$state_dict(),
    tolerance = 1e-6
  )

  latest <- torch::torch_load(resumed_path)
  best <- torch::torch_load(fit_resumed$best_checkpoint_path)
  expect_equal(latest$epoch, 4L)
  expect_equal(best$epoch, fit_resumed$best_val_epoch)
  expect_null(latest$schema_version)
  expect_null(best$schema_version)
  expect_false(identical(resumed_path, fit_resumed$best_checkpoint_path))
  expect_identical(
    predict_mst_pmdn(fit_resumed$model, x[1:2, , drop = FALSE])$skew_none,
    TRUE
  )
})
