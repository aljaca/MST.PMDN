################################################################################
##  Comparison of mclust vs. PMDN (multivariate normal) on the iris data set
################################################################################

rm(list = ls())

library(ggplot2)
library(GGally)
library(ellipse)
library(mclust)
library(torch)
source("../R/MST-PMDN.R")
torch_set_num_threads(1)

data(iris)
X  <- as.matrix(scale(iris[, 1:4]))
G  <- 3
N  <- nrow(X)

mclust_models <- c(
  "EII", "VII", "EEI", "VEI", "EVI", "EEE", "VVI",
  "VEE", "EVE", "VVE", "EEV", "VEV", "EVV", "VVV"
)

fit_mclust <- function(x, model, G) {
  tryCatch(
    {
      fit <- Mclust(x, G = G, modelNames = model, verbose = FALSE)
      list(loglik = as.numeric(logLik(fit)),
           parameters = fit$parameters)
    },
    error = function(e) list(loglik = NA, parameters = NA)
  )
}

fit_pmdn <- function(x, model, G, N, epochs = 500) {
  constraint     <- paste0(model, "FN")
  constant_attr  <- "LADmxns"
  fixed_nu <- rep(1e6, G)
  fit <- train_mst_pmdn(
    inputs           = matrix(NA, nrow = N, ncol = 0),
    outputs          = x,
    hidden_dim       = integer(0),
    n_mixtures       = G,
    constraint       = constraint,
    constant_attr    = constant_attr,
    epochs           = epochs,
    fixed_nu         = fixed_nu,
    range_nu         = c(3, 1e6),
    lr               = 1e-2,
    batch_size       = 10,
    validation_split = 0.
  )
  negloglik <- tail(fit$best_train_loss, 1) * N
  list(negloglik = negloglik, fit = fit)
}

results <- data.frame(Model            = character(),
                      Framework        = character(),
                      NegLogLik        = double(),
                      stringsAsFactors = FALSE)

fits_mc <- fits_pm <- list()
for (m in mclust_models) {
  ## mclust
  mc <- fit_mclust(X, m, G)
  fits_mc[[m]] <- mc$parameters
  results <- rbind(results,
                   data.frame(Model     = m,
                              Framework = "mclust",
                              NegLogLik = -mc$loglik))

  ## MST-PMDN
  if (!is.na(mc$loglik)) {
    pm <- fit_pmdn(X, m, G, N)
    fits_pm[[m]] <- pm$fit
    results <- rbind(results,
                     data.frame(Model     = m,
                                Framework = "MST-PMDN",
                                NegLogLik = pm$negloglik))
  }
}

##
# Results

nloglik_mixture <- function(X, pi, mu, Sig) {
  G  <- length(pi)
  n  <- nrow(X)

  dens <- matrix(0, n, G)
  for (k in seq_len(G)) {
    dens[, k] <- dmvnorm(X, mean = mu[, k], sigma = Sig[, , k], log = FALSE)
  }
  f  <- dens %*% pi
  -sum(log(f))
}

NLL <- as.data.frame(matrix(NA, ncol = 2, nrow = length(mclust_models)))
colnames(NLL) <- c("mclust", "PMDN")
rownames(NLL) <- mclust_models
for (m in mclust_models) {
  pi_mc <- fits_mc[[m]]$pro
  mu_mc <- fits_mc[[m]]$mean
  Sig_mc <- fits_mc[[m]]$variance$sigma
  NLL_mc <- nloglik_mixture(X, pi_mc, mu_mc, Sig_mc)

  out <- fits_pm[[m]]$model$forward(torch_tensor(matrix(1, 1, 1)))
  pi_pm <- as.numeric(out$pi)
  mu_pm <- t(as.matrix(out$mu[1, , ]))
  G <- length(pi_pm)
  d <- nrow(mu_pm)
  Sig_pm <- array(NA_real_, dim = c(d, d, G))
  for (k in seq_len(G)) {
    Lk <- as.matrix(out$scale_chol[1, k, , ])
    Sig_pm[, , k] <- Lk %*% t(Lk)
  }
  NLL_pm <- nloglik_mixture(X, pi_pm, mu_pm, Sig_pm)

  NLL[m, "mclust"] <- NLL_mc
  NLL[m, "PMDN"] <- NLL_pm
}
print(NLL)

pdf(file = "example-iris-mclust.pdf", width = 8, height = 8)

par(mar = c(4, 4, 2, 1))
names <- rep(mclust_models, each = 2)
names[c(FALSE, TRUE)] <- ""
cols <- c("lightblue", "blue")
barplot(unlist(t(NLL)), beside = TRUE, col = cols,
        names = names, ylab = "NLL", ylim = c(0, max(pretty(unlist(c(NLL))))))
legend("topright", c("mclust", "PMDN"), col = cols, pch = 15, pt.cex = 2)
box()

##
# Plot samples from fitted mclust and PMDN models

n_synth <- 1000
m <- "VEV"

sim_mc <- sim(modelName = m, parameters = fits_mc[[m]], n = n_synth)
sim_mc <- as.data.frame(sim_mc[, c(2:(ncol(X) + 1), 1)])

pred_pm <- predict_mst_pmdn(model = fits_pm[[m]]$model,
                            new_inputs = matrix(NA, nrow = n_synth, ncol = 0))
sim_pm <- sample_mst_pmdn_df(pred_pm, num_samples = 1)
sim_pm <- as.data.frame(sim_pm[, c(seq_len(ncol(X)), ncol(sim_pm))])

names(sim_mc) <- names(sim_pm) <- c(colnames(X), "z")
sim_mc[, "z"] <- as.factor(sim_mc[, "z"])
sim_pm[, "z"] <- as.factor(sim_pm[, "z"])

iris_scaled <- as.data.frame(cbind(scale(iris[, seq_len(ncol(X))]),
                                   iris[, ncol(X) + 1]))
names(iris_scaled) <- c(colnames(X), "z")
iris_scaled[, "z"] <- as.factor(iris_scaled[, "z"])

pairplot_mix <- function(df, title) {
  ggpairs(df, aes(colour = z, alpha = 0.5),
          columns = 1:4,
          lower = list(continuous = wrap("points", size = 0.7)),
          upper = list(continuous = function(data, mapping, ...) {
            ggplot(data = data, mapping = mapping) +
              geom_point(size = 0.35, alpha = 0.25) +
              stat_ellipse(level = 0.50, linewidth = 1, alpha = 0.8) +
              stat_ellipse(level = 0.95, linewidth = 0.75, alpha = 0.8)
          }),
          diag  = list(continuous = "densityDiag")) +
    ggtitle(title) +
    theme_bw(base_size = 9)
}

plot_iris <- pairplot_mix(iris_scaled, "iris dataset")
plot_mc <- pairplot_mix(sim_mc, paste0("mclust simulated components (", m, ")"))
plot_pm <- pairplot_mix(sim_pm, paste0("PMDN simulated components (", m, ")"))

print(plot_iris)
print(plot_mc)
print(plot_pm)

dev.off()

################################################################################
