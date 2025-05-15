################################################################################
# Multivariate skew t-Parsimonious Mixture Density Network (MST-PMDN)
# Wave-surge example (CCCRIS node 181947; Roberts Bank Superport)

rm(list = ls())
set.seed(1)

library(mclust)
library(MASS)
library(ncdf4)
library(scales)
library(abind)
source("../MST-PMDN.R")
torch_set_num_threads(1)

##
# Tabular data (lag-1 wave-surge and first two harmonics of the annual cycle

data <- read.csv("CCCRIS-181947_wave-surge_ERA5.csv")

y <- data[, c("Wave.m", "Surge.m")]
y <- apply(y, 2, function(x) scale(jitter(x, 5)))

y_lag1 <- y[-nrow(y), ]
y <- y[-1, ]

date <- data[-1, 1]
custom_split <- substr(date, 1, 4) <= 2015

doy <- as.numeric(format(as.Date(date), "%j"))
doy_sc <- c()
for (i in 1:2) {
  doy_sc <- cbind(doy_sc, sin(2 * pi * i * doy / 365.25),
                  cos(2 * pi * i * doy / 365.25))
}

x <- scale(cbind(doy_sc, y_lag1))

##
# Image data (mean sea level pressure and sea level pressure gradient; 64 x 64)

nc <- nc_open("CCCRIS-181947_psl_ERA5.nc")
psl <- ncvar_get(nc, varid = "psl")
nc_close(nc)

nc <- nc_open("CCCRIS-181947_psl-grad_ERA5.nc")
psl_grad <- ncvar_get(nc, varid = "psl_grad")
nc_close(nc)

psl_mean <- apply(psl, c(1, 2), mean)
psl_sd <- apply(psl, c(1, 2), sd)
psl <- sweep(sweep(psl, c(1, 2), psl_mean, "-"), c(1, 2), psl_sd, "/")

psl_grad_mean <- apply(psl_grad, c(1, 2), mean)
psl_grad_sd <- apply(psl_grad, c(1, 2), sd)
psl_grad <- sweep(sweep(psl_grad, c(1, 2), psl_grad_mean, "-"), c(1, 2),
                  psl_grad_sd, "/")

x_image <- abind(psl, psl_grad, along = -1)

# Reshape [time x channels x lon x lat]
x_image <- aperm(x_image, c(4, 1, 2, 3))
x_image <- x_image[-1, , , ]

rm(psl, psl_grad)

##
# Tabular module

tabular_module <- nn_module(
  "TabularModule",
  initialize = function(input_dim, hidden_dims, output_dim, dropout_rate) {
    if (is.null(hidden_dims) || length(hidden_dims) == 0) {
      self$n_hidden_layers <- 0
      self$hidden_dims <- c()
    } else if (!is.vector(hidden_dims) && !is.list(hidden_dims)) {
      self$hidden_dims <- c(hidden_dims)
      self$n_hidden_layers <- length(self$hidden_dims)
    } else {
      self$hidden_dims <- hidden_dims
      self$n_hidden_layers <- length(self$hidden_dims)
    }
    self$output_dim <- output_dim
    self$dropout_rate <- dropout_rate

    self$layers <- nn_module_list()
    self$bns <- nn_module_list()
    if (self$dropout_rate > 0) {
        self$dropouts <- nn_module_list()
    }
    current_dim <- input_dim
    if (self$n_hidden_layers > 0) {
      for (i in 1:self$n_hidden_layers) {
        self$layers$append(nn_linear(current_dim, self$hidden_dims[[i]]))
        self$bns$append(nn_batch_norm1d(self$hidden_dims[[i]]))
        if (self$dropout_rate > 0) {
            self$dropouts$append(nn_dropout(p = self$dropout_rate))
        }
        current_dim <- self$hidden_dims[[i]]
      }
    }
    self$layers$append(nn_linear(current_dim, output_dim))
  },
  forward = function(x) {
    for (i in 1:self$n_hidden_layers) {
      x <- self$layers[[i]](x)
      x <- self$bns[[i]](x)
      x <- nnf_relu(x)
      if (self$dropout_rate > 0 && !is.null(self$dropouts[[i]])) {
          x <- self$dropouts[[i]](x)
      }
    }
    x <- self$layers[[length(self$layers)]](x)
    x <- nnf_relu(x)
    x
  }
)

tabular_mod <- tabular_module(
  input_dim = ncol(x),
  hidden_dims = c(32, 16),
  output_dim = 16,
  dropout_rate = 0.1
)

##
# Image module

image_module <- nn_module(
  initialize = function() {
    self$conv1 <- nn_conv2d(in_channels = 2, out_channels = 16,
                            kernel_size = 3, padding = 1)
    self$bn1 <- nn_batch_norm2d(num_features = 16)
    self$pool1 <- nn_max_pool2d(kernel_size = 2)
    self$conv2 <- nn_conv2d(in_channels = 16, out_channels = 32,
                            kernel_size = 3, padding = 1)
    self$bn2 <- nn_batch_norm2d(num_features = 32)
    self$pool2 <- nn_max_pool2d(kernel_size = 2)
    self$flatten_dim <- 32 * 16 * 16
    self$fc <- nn_linear(self$flatten_dim, 32)
    self$bn_fc <- nn_batch_norm1d(num_features = 32)
    self$output_dim <- 32
  },
  forward = function(x) {
    x <- self$conv1(x)
    x <- self$bn1(x)
    x <- nnf_relu(x)
    x <- self$pool1(x)
    x <- self$conv2(x)
    x <- self$bn2(x)
    x <- nnf_relu(x)
    x <- self$pool2(x)
    x <- torch_flatten(x, start_dim = 2)
    x <- self$fc(x)
    x <- self$bn_fc(x)
    nnf_relu(x)
  }
)

image_mod <- image_module()

##
# Fusion MLP

hidden_dim <- c(64, 32)

## MST-MDN head
# Volume (L)-Shape (A)-Orientation (D) and MST constraints

modelname <- "VEV"
skewtname <- "FN"
constant_attr <- ""
n_mixtures <- 4
fixed_nu <- c(rep(50, n_mixtures - 1), NA)

##
# Model-based clustering (mclust) on independent samples

print(Mclust(y[which(custom_split)[c(TRUE, rep(FALSE, 5))], ], G = 1:10)$BIC)

##
# Training/validation using tabular and image inputs

t1 <- Sys.time()
fit <- train_mst_pmdn(
  inputs = x,
  outputs = y,
  hidden_dim = hidden_dim,
  n_mixtures = n_mixtures,
  constraint = paste0(modelname, skewtname),
  constant_attr = constant_attr,
  fixed_nu = fixed_nu,
  activation = nn_relu,
  range_nu = c(3., 50.),
  max_alpha = 5.,
  min_vol_shape = 0.01,
  jitter = 1e-4,
  lr = 0.0001,
  max_norm = 1.,
  epochs = 100,
  batch_size = 16,
  drop_hidden = 0.1,
  wd_image = 1e-3,
  wd_tabular = 0.,
  checkpoint_interval = 10,
  checkpoint_path = "wave-surge-checkpoint.pt",
  resume_from_checkpoint = FALSE,
  early_stopping_patience = 20,
  validation_split = 0,
  custom_split = custom_split,
  scheduler_step = 50,
  scheduler_gamma = 0.5,
  image_inputs = x_image,
  image_module = image_mod,
  tabular_module = tabular_mod
)

t2 <- Sys.time()
print(t2 - t1)

##
# Prediction using both tabular and image inputs.

pred <- predict_mst_pmdn(fit$model, x, x_image, device = "cpu")
samples <- as.matrix(sample_mst_pmdn(pred, num_samples = 1)$samples[1, , ])
samples[samples[, 1] < min(y[, 1]), 1] <- min(y[, 1])
cat("mu:\n")
print(pred$mu[1:2, , ])
cat("pi:\n")
print(pred$pi[1:2, ])
cat("L:\n")
print(pred$L[1:2, , ])
cat("A:\n")
print(pred$A[1:2, , ])
cat("D:\n")
print(pred$D[1:2, , ])
cat("nu:\n")
print(pred$nu[1:2, ])
cat("alpha:\n")
print(pred$alpha[1:2, , ])

##
# Stochastic samples (lag-1 autoregressive terms)

rsamples <- matrix(0, ncol = 2, nrow = nrow(x))
rsamples[1, ] <- y[1, ]
for (i in 2:nrow(x)) {
  if (i %% 100 == 0) cat(i, "")
  x_i <- cbind(x[i, 1:(ncol(x) - 2), drop = FALSE],
               rsamples[i - 1, , drop = FALSE])
  x_image_i <- x_image[i , , , , drop=FALSE]
  rsamples_i <- sample_mst_pmdn(predict_mst_pmdn(fit$model, x_i, x_image_i,
                                                 device = "cpu"),
                                num_samples = 1)$samples[1, , ]
  rsamples[i, ] <- as.matrix(rsamples_i)
}
rsamples[rsamples[, 1] < min(y[, 1]), 1] <- min(y[, 1])
cat("\n")

##
# Compare with model-based clustering

mc <- Mclust(y, G = n_mixtures, modelNames = modelname)
print(mc$df)
print(mc$loglik)
mc_samples <- sim(modelName = mc$modelName,
                  parameters = mc$parameters,
                  n = nrow(y))[, -1]
mc_samples[mc_samples[, 1] < min(y[, 1]), 1] <- min(y[, 1])

colnames(y) <- colnames(samples) <- colnames(rsamples) <-
  colnames(mc_samples) <- c("Z[Wave+1]", "Z[Surge+1]")

##
# Plot data and random samples

pdf("wave-surge-dailymax.pdf", width = 12, height = 8)

pch <- 19
cex <- 1.5
par(mfrow = c(2, 3), mar = c(4.5, 4.5, 3.5, 1))
plot(y[, 1], y[, 2], main = "Original Data", col = alpha("black", 0.01),
     pch = pch, cex = cex, xlab = colnames(y)[1], ylab = colnames(y)[2])
plot(rsamples[, 1], rsamples[, 2], xlab = colnames(y)[1], ylab = colnames(y)[2],
     main = "MST-PMDN samples", col = alpha("darkblue", 0.01), pch = pch,
     cex = cex)
plot(mc_samples[, 1], mc_samples[, 2], xlab = colnames(y)[1],
     ylab = colnames(y)[2], main = "mclust samples", col = alpha("red", 0.01),
     pch = pch, cex = cex)
image(MASS::kde2d(y[, 1], y[, 2]))
image(MASS::kde2d(rsamples[, 1], rsamples[, 2]))
image(MASS::kde2d(mc_samples[, 1], mc_samples[, 2]))

acf(y, col = "black", lwd = 3)
acf(rsamples, col = "darkblue", lwd = 4)
acf(mc_samples, col = "red", lwd = 4)

par(mfrow = c(1, 1))
hist(as.numeric(pred$nu[, ncol(pred$nu)]), border = NA, xlab = "nu",
     main = "MST-PMDN component with\nvariable degrees of freedom",
     freq = FALSE, col = "darkblue")
box()
abline(v = 30)

par(mfrow = c(1, 2))
qqplot(y[, 1], rsamples[, 1], xlab = "y1", ylab = "rsamples1")
abline(0, 1, col = "red", lty = 2, lwd = 2)
grid()
qqplot(y[, 2], rsamples[, 2], xlab = "y2", ylab = "rsamples2")
abline(0, 1, col = "red", lty = 2, lwd = 2)
grid()

dev.off()

cat("Validation statistics:\n")
print(cor(y[!custom_split, ]))
print(cor(mc_samples[!custom_split, ]))
print(cor(rsamples[!custom_split, ]))

print(cor(x[!custom_split, ], y[!custom_split, ]))
print(cor(y[!custom_split, ], mc_samples[!custom_split, ]))
print(cor(y[!custom_split, ], rsamples[!custom_split, ]))

################################################################################
