################################################################################
# Multivariate skew t-Parsimonious Mixture Density Network (MST-PMDN) 
# Wave-surge example

rm(list=ls())
library(mclust)
library(MASS)
library(scales)
source("../MST-PMDN.R")

data <- read.csv("wave+surge_CCCRIS_181947.csv")
data <- data[c(TRUE, rep(FALSE, 5)),] # 1-hr -> sub-samples

y <- data[,c("Wave.m", "Surge.m")]
y <- log(y + 1)
y <- apply(y, 2, function(x) scale(jitter(x)))

y.lag1 <- y[-nrow(y),]
y <- y[-1,]

date <- data[-1,"Date"]
time <- data[-1,"Time"]

doy <- as.numeric(format(as.Date(date), "%j"))
hr <- as.numeric(substr(time, 1, 2))

doy.sc <- hr.sc <- c()
for(i in 1:2){
  doy.sc <- cbind(doy.sc, sin(2*pi*i*doy/365.25),
                  cos(2*pi*i*doy/365.25))
  hr.sc <- cbind(hr.sc, sin(2*pi*i*hr/24),
                 cos(2*pi*i*hr/24))
}

x <- scale(cbind(doy.sc, hr.sc, y.lag1))

# Training/validation split
custom_split <- substr(date, 1, 4) <= 2015

# volume (L)-shape (A)-orientation (D)
modelName <- "VVV"
skewtname <- "FN"
constant_attr <- ""
n_mixtures <- 4
fixed_nu <-c(rep(50, n_mixtures-1), NA)

tabular_module <- nn_module(
  "TabularModule",
  initialize = function(input_dim, hidden_dim, output_dim) {
    self$fc1 <- nn_linear(input_dim, hidden_dim)
    self$fc2 <- nn_linear(hidden_dim, output_dim)
    self$output_dim <- output_dim
  },
  forward = function(x) {
    x <- nnf_relu(self$fc1(x))
    x <- self$fc2(x)
    x
  }
)
tabular_mod <- tabular_module(input_dim = ncol(x), hidden_dim = 5,
                              output_dim = 5)

t1 <- Sys.time()
fit <- train_mst_pmdn(
  inputs = x,
  outputs = y,
  hidden_dim = 3,
  n_mixtures = n_mixtures,
  constraint = paste0(modelName, skewtname),
  constant_attr = constant_attr,
  fixed_nu = fixed_nu,
  activation = nn_relu,
  range_nu = c(3., 50.),
  max_alpha = 5.,
  min_vol_shape = 1e-3,
  jitter = 1e-1,
  lr = 0.0001,
  epochs = 200,
  batch_size = 16,
  wd_hidden = 1e-3,
  checkpoint_interval = 10,
  checkpoint_path = "mdn_checkpoint.pt",
  resume_from_checkpoint = FALSE,
  early_stopping_patience = 20,
  validation_split = 0,
  custom_split = custom_split,
  scheduler_step = 50,
  scheduler_gamma = 0.5,
  image_inputs = NULL,
  image_module = NULL,
  tabular_module = tabular_mod
)

t2 <- Sys.time()
print(t2-t1)

# Prediction using both structured and image inputs.
pred <- predict_mst_pmdn(fit$model, x, device = "cpu")
samples <- as.matrix(sample_mst_pmdn(pred, num_samples = 1)[1,,])
samples[samples[,1] < min(y[,1]),1] <- min(y[,1])
cat("mu:\n")
print(pred$mu[1:2,,])
cat("pi:\n")
print(pred$pi[1:2,])
cat("L:\n")
print(pred$L[1:2,,])
cat("A:\n")
print(pred$A[1:2,,])
cat("D:\n")
print(pred$D[1:2,,])
cat("nu:\n")
print(pred$nu[1:2,])
cat("alpha:\n")
print(pred$alpha[1:2,,])

# Stochastic samples (lag-1 autoregressive terms) 
rsamples <- matrix(0, ncol=2, nrow=nrow(x))
rsamples[1,] <- y[1,]
for(i in 2:nrow(x)){
  if(i%%1000==0) cat(i, "")
  x.i <- cbind(x[i,1:(ncol(x)-2),drop=FALSE],rsamples[i-1,,drop=FALSE])
  rsamples.i <- sample_mst_pmdn(predict_mst_pmdn(fit$model, x.i,
                                  device = "cpu"), num_samples = 1)[1,,]
  rsamples[i,] <- as.matrix(rsamples.i)
}
rsamples[rsamples[,1] < min(y[,1]),1] <- min(y[,1])
cat("\n")

# Compare with model-based clustering
mc <- Mclust(y, G=n_mixtures, modelNames=modelName)
print(mc$df)
print(mc$loglik)
mc.samples <- sim(modelName=mc$modelName,
                  parameters=mc$parameters,
                  n=nrow(y))[,-1]
mc.samples[mc.samples[,1] < min(y[,1]),1] <- min(y[,1])

colnames(y) <- colnames(samples) <- colnames(rsamples) <-
  colnames(mc.samples) <- c("Z[logWave+1]", "Z[logSurge+1]")

# Plot data and random samples
pch <- 19
cex <- 1.5
par(mfrow=c(2, 3), mar=c(4.5, 4.5, 3.5, 1))
plot(y[,1], y[,2], main="Original Data", col=alpha("black", 0.01),
     pch=pch, cex=cex, xlab=colnames(y)[1], ylab=colnames(y)[2])
plot(rsamples[,1], rsamples[,2], xlab=colnames(y)[1], ylab=colnames(y)[2],
     main="MST-PMDN samples", col=alpha("darkblue", 0.01), pch=pch, cex=cex)
plot(mc.samples[,1], mc.samples[,2], xlab=colnames(y)[1], ylab=colnames(y)[2],
     main="mclust samples", col=alpha("red", 0.01), pch=pch,
     cex=cex)
image(MASS::kde2d(y[,1], y[,2]))
image(MASS::kde2d(rsamples[,1], rsamples[,2]))
image(MASS::kde2d(mc.samples[,1], mc.samples[,2]))

acf(y, col="black", lwd=3)
acf(rsamples, col="darkblue", lwd=4)
acf(mc.samples, col="red", lwd=4)

par(mfrow=c(1, 1))
hist(as.numeric(pred$nu[,ncol(pred$nu)]), border=NA, xlab="nu",
     main="MST-PMDN component with\nvariable degrees of freedom", freq=FALSE,
     col="darkblue")
box()
abline(v=30)

print(cor(y))
print(cor(mc.samples))
print(cor(rsamples))

print(cor(x, y))
print(cor(y, mc.samples))
print(cor(y, rsamples))

print(Mclust(y, G=1:8)$BIC)

################################################################################
