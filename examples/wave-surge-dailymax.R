################################################################################
# Multivariate skew-t Parsimonious Mixture Density Network (MST-PMDN)
# Wave-surge example: CCCRIS node 181947, Roberts Bank Superport
#
# Predictors
#   - first two annual sine/cosine harmonics
#   - lag-1, lag-2, and lag-3 wave height and surge
#   - daily mean sea-level pressure and 10-m U/V winds on a 64 x 64 grid
#
# Cold seasons are October-March and are labelled by the year in which March
# occurs. Training, validation, and test winters are 1981-2015, 2016-2018,
# and 2019, respectively.
################################################################################

library(torch)
library(MST.PMDN)

seed <- 1747892421L
set.seed(seed)
torch_manual_seed(seed)

device <- if (cuda_is_available()) "cuda" else "cpu"
torch_set_num_threads(1L)
torch_set_num_interop_threads(1L)

# Preprocessed wave_surge data
data_file <- "wave_surge.RData"

# Output files
output_dir <- "out"
if (!dir.exists(output_dir) &&
    !dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)) {
  stop("Could not create output directory: ", output_dir)
}
checkpoint_file <- file.path(output_dir, "wave-surge-dailymax-checkpoint.pt")
model_file <- file.path(output_dir, "wave-surge-dailymax.pt")
results_file <- file.path(output_dir, "wave-surge-dailymax-results.rds")

##
# Load the preprocessed wave_surge data

load(data_file)
model_date <- as.Date(wave_surge$date)
y <- wave_surge$y
y_original <- wave_surge$y_original
x <- wave_surge$x
x_image <- wave_surge$x_image
y_mean <- wave_surge$y_mean
y_sd <- wave_surge$y_sd
x_image_mean <- wave_surge$x_image_mean
x_image_sd <- wave_surge$x_image_sd
wave_scale <- wave_surge$wave_scale
inv_softplus <- wave_surge$inv_softplus
softplus <- wave_surge$softplus

# Number of lags and harmonics in tabular covariates.
ar_lags <- seq(sum(grepl("lag", colnames(wave_surge$x)))/2)
max_lag <- max(ar_lags)
n_harmonics <- sum(grepl("cos", colnames(wave_surge$x)) |
  grepl("sin", colnames(wave_surge$x)))

# October-December belong to the winter ending in the following calendar year.
month <- as.integer(format(model_date, "%m"))
year <- as.integer(format(model_date, "%Y"))
model_season <- year + as.integer(month >= 10)
train <- model_season >= 1981 & model_season <= 2015
validation <- model_season >= 2016 & model_season <= 2018
test <- model_season == 2019

cat(
  "Rows: training =", sum(train),
  "; validation =", sum(validation),
  "; test =", sum(test), "\n"
)
cat(
  "Test winter:", format(min(model_date[test])), "to",
  format(max(model_date[test])), "\n"
)

##
# Predictor modules
#
# The tabular branch is deliberately small because it has only ten inputs.
# The CNN uses GroupNorm for stable small-batch training and retains a 4 x 4
# spatial layout before its dense projection; global pooling would incorrectly
# make a storm pattern independent of its position relative to Roberts Bank.

tabular_module <- nn_module(
  "WaveSurgeTabularModule",
  initialize = function(input_dim, output_dim = 32, dropout_rate = 0.2) {
    self$output_dim <- output_dim
    self$fc1 <- nn_linear(input_dim, 64)
    self$norm1 <- nn_layer_norm(64)
    self$fc2 <- nn_linear(64, output_dim)
    self$norm2 <- nn_layer_norm(output_dim)
    self$dropout <- nn_dropout(p = dropout_rate)
  },
  forward = function(x) {
    x <- nnf_gelu(self$norm1(self$fc1(x)))
    x <- self$dropout(x)
    nnf_gelu(self$norm2(self$fc2(x)))
  }
)

image_module <- nn_module(
  "WaveSurgeImageModule",
  initialize = function(in_channels, output_dim = 64, dropout_rate = 0.2) {
    self$output_dim <- output_dim
    self$conv1 <- nn_conv2d(
      in_channels, 16, kernel_size = 5, stride = 2, padding = 2
    )
    self$conv2 <- nn_conv2d(
      16, 32, kernel_size = 3, stride = 2, padding = 1
    )
    self$conv3 <- nn_conv2d(
      32, 64, kernel_size = 3, stride = 2, padding = 1
    )
    self$conv4 <- nn_conv2d(
      64, 96, kernel_size = 3, stride = 2, padding = 1
    )
    self$norm1 <- nn_group_norm(8, 16)
    self$norm2 <- nn_group_norm(8, 32)
    self$norm3 <- nn_group_norm(8, 64)
    self$norm4 <- nn_group_norm(8, 96)
    self$pool <- nn_adaptive_avg_pool2d(output_size = c(4, 4))
    self$fc <- nn_linear(96 * 4 * 4, output_dim)
    self$fc_norm <- nn_layer_norm(output_dim)
    self$dropout <- nn_dropout(p = dropout_rate)
  },
  forward = function(x) {
    x <- nnf_gelu(self$norm1(self$conv1(x)))
    x <- nnf_gelu(self$norm2(self$conv2(x)))
    x <- nnf_gelu(self$norm3(self$conv3(x)))
    x <- nnf_gelu(self$norm4(self$conv4(x)))
    x <- self$pool(x)
    x <- torch_flatten(x, start_dim = 2)
    x <- self$dropout(x)
    nnf_gelu(self$fc_norm(self$fc(x)))
  }
)

# Concatenation followed by a compact nonlinear projection allows the two
# branches to contribute complementary information without forcing them into a
# convex gate.
fusion_module <- nn_module(
  "WaveSurgeFusionModule",
  initialize = function(
    tabular_dim,
    image_dim,
    output_dim = 48,
    dropout_rate = 0.2
  ) {
    self$output_dim <- output_dim
    self$fc1 <- nn_linear(tabular_dim + image_dim, 64)
    self$norm1 <- nn_layer_norm(64)
    self$fc2 <- nn_linear(64, output_dim)
    self$norm2 <- nn_layer_norm(output_dim)
    self$dropout <- nn_dropout(p = dropout_rate)
  },
  forward = function(tabular_features, image_features) {
    x <- torch_cat(list(tabular_features, image_features), dim = 2)
    x <- nnf_gelu(self$norm1(self$fc1(x)))
    x <- self$dropout(x)
    nnf_gelu(self$norm2(self$fc2(x)))
  }
)

tabular_mod <- tabular_module(
  input_dim = ncol(x),
  output_dim = 32,
  dropout_rate = 0.1
)
image_mod <- image_module(
  in_channels = dim(x_image)[2],
  output_dim = 64,
  dropout_rate = 0.1
)
fusion_mod <- fusion_module(
  tabular_dim = tabular_mod$output_dim,
  image_dim = image_mod$output_dim,
  output_dim = 48,
  dropout_rate = 0.1
)

##
# MST-PMDN training
#

n_mixtures <- 1L
constraint <- "VVVVV"
fixed_nu <- c()
constant_attr <- ""

x_test <- x[test, , drop = FALSE]
x_image_test <- x_image[test, , , , drop = FALSE]
y_test_original <- y_original[test, , drop = FALSE]
date_test <- model_date[test]

development <- train | validation
x_development <- x[development, , drop = FALSE]
x_image_development <- x_image[development, , , , drop = FALSE]
y_development <- y[development, , drop = FALSE]
custom_split <- train[development]

t1 <- Sys.time()
fit <- train_mst_pmdn(
  inputs = x_development,
  outputs = y_development,
  hidden_dim = integer(0),
  n_mixtures = n_mixtures,
  constraint = constraint,
  constant_attr = constant_attr,
  fixed_nu = fixed_nu,
  range_nu = c(3, 50),
  max_alpha = 5,
  min_vol_shape = 0.01,
  jitter = 1e-5,
  activation = nn_gelu,
  epochs = 300L,
  lr = 3e-4,
  batch_size = 32L,
  max_norm = 1,
  drop_hidden = 0.2,
  wd_image = 1e-3,
  wd_tabular = 1e-3,
  checkpoint_interval = 10L,
  checkpoint_path = checkpoint_file,
  resume_from_checkpoint = FALSE,
  early_stopping_patience = 15L,
  validation_split = 0,
  custom_split = custom_split,
  scheduler_step = 10L,
  scheduler_gamma = 0.7,
  image_inputs = x_image_development,
  image_module = image_mod,
  tabular_module = tabular_mod,
  fusion_module = fusion_mod,
  device = device
)
cat(device, " training time:", format(Sys.time() - t1), "\n")
torch_save(fit$model, model_file)

##
# One-step-ahead inference

pred_test <- predict_mst_pmdn(
  fit$model,
  new_inputs = x_test,
  image_inputs = x_image_test,
  device = device
)
cat("First three mixture weights:\n")
print(as.array(pred_test$pi[1:3, ]$to(device = "cpu")))
cat("First three degrees of freedom:\n")
print(as.array(pred_test$nu[1:3, ]$to(device = "cpu")))

##
# Recursive cold-season ensemble
#
# The first prediction is 4 October 2018. Its saved lag predictors contain the
# observed standardized values from 1-3 October. Thereafter, generated values
# remain on the standardized model scale while being fed back as lags;
# back-transformation is performed only after the full recursive simulation.

n_ens <- 49L
n_test <- nrow(x_test)
if (!identical(format(date_test[1]), "2018-10-04")) {
  stop("The saved test predictors must begin on 2018-10-04.")
}

lag_state <- lapply(seq_along(ar_lags), function(j) {
  lag_columns <- n_harmonics +
    (j - 1L) * ncol(y) + seq_len(ncol(y))
  matrix(
    rep(x_test[1, lag_columns], each = n_ens),
    nrow = n_ens,
    ncol = ncol(y)
  )
})

samples_scaled <- array(
  NA_real_,
  dim = c(n_test, n_ens, ncol(y)),
  dimnames = list(
    date = format(date_test),
    member = seq_len(n_ens),
    variable = colnames(y)
  )
)

for (i in seq_len(n_test)) {
  harmonic_i <- matrix(
    rep(x_test[i, seq_len(n_harmonics)], each = n_ens),
    nrow = n_ens,
    ncol = n_harmonics
  )
  x_i <- cbind(harmonic_i, do.call(cbind, lag_state))
  x_image_i <- x_image_test[rep(i, n_ens), , , , drop = FALSE]

  pred_i <- predict_mst_pmdn(
    fit$model,
    new_inputs = x_i,
    image_inputs = x_image_i,
    device = device
  )
  draw_i <- sample_mst_pmdn(
    pred_i,
    num_samples = 1L,
    device = device
  )$samples[1, , ]
  y_i <- as.matrix(as.array(draw_i$to(device = "cpu")))

  samples_scaled[i, , ] <- y_i
  lag_state <- c(list(y_i), lag_state[seq_len(max_lag - 1L)])

  if (i %% 10L == 0L || i == n_test) {
    cat("Recursive simulation:", i, "of", n_test, "days\n")
  }
}

samples_original <- samples_scaled
wave_transformed <- samples_scaled[, , 1] * y_sd[1] + y_mean[1]
samples_original[, , 1] <- softplus(wave_transformed, wave_scale)
samples_original[, , 2] <- (
  samples_scaled[, , 2] * y_sd[2] + y_mean[2]
)
dimnames(samples_original)[[3]] <- colnames(y_original)

energy_score <- vapply(seq_len(n_test), function(i) {
  scoringRules::es_sample(
    y = y_test_original[i, ],
    dat = t(samples_original[i, , ])
  )
}, numeric(1))
ensemble_correlation <- vapply(seq_len(n_ens), function(j) {
  cor(samples_original[, j, ])[1, 2]
}, numeric(1))

cat("Mean recursive test energy score:", mean(energy_score), "\n")
cat("Observed test correlation:", cor(y_test_original)[1, 2], "\n")
cat(
  "Mean ensemble-member test correlation:",
  mean(ensemble_correlation), "\n"
)

##
# Save compact reproducibility/evaluation data

for (i in names(pred_test)) {
  pred_test[[i]] <- as.array(pred_test[[i]])
}

saveRDS(
  list(
    seed = seed,
    constraint = constraint,
    fixed_nu = fixed_nu,
    n_mixtures = n_mixtures,
    ar_lags = ar_lags,
    train_seasons = 1981:2015,
    validation_seasons = 2016:2018,
    test_season = 2019L,
    test_dates = date_test,
    wave_transform_scale = wave_scale,
    target_center = y_mean,
    target_scale = y_sd,
    observed_test = y_test_original,
    pred_test = pred_test,
    recursive_samples = samples_original,
    energy_score = energy_score,
    ensemble_correlation = ensemble_correlation,
    train_loss_history = fit$train_loss_history,
    val_loss_history = fit$val_loss_history
  ),
  results_file
)

################################################################################
