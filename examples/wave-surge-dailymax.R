################################################################################
# Multivariate skew t-Parsimonious Mixture Density Network (MST-PMDN)
# Wave-surge example (CCCRIS node 181947; Roberts Bank Superport)
# Uses: torch, ncdf4, abind, mclust, MASS, scales, abind, scoringRules, ddalpha

seed <- 1747892421
set.seed(seed)
print(seed)

library(mclust)
library(MST.PMDN)

device <- ifelse(cuda_is_available(), "cuda", "cpu")
num_threads <- 1
torch_set_num_threads(num_threads)
torch_set_num_interop_threads(num_threads)

##
# Tabular data (lag-1 wave-surge and first two harmonics of the annual cycle

data <- read.csv("CCCRIS-181947_wave-surge_ERA5.csv")
date <- data[-1, 1]

# Training (1980-2015) and validation (2016-2019) splits 
custom_split <- substr(date, 1, 4) <= 2015

y <- data[, c("Wave.m", "Surge.m")]
y <- jitter(as.matrix(y), 5)

# Scale targets based on training split statistics
y_mean <- apply(y[custom_split, ], 2, mean)
y_sd <- apply(y[custom_split, ], 2, sd)
y <- scale(y, center = y_mean, scale = y_sd)
y1_min <- min(y[custom_split, 1])

# Teacher forcing (lag-1 autocorrelation) and harmonics of the annual cycle
y_lag1 <- y[-nrow(y), ]
y <- y[-1, ]

doy <- as.numeric(format(as.Date(date), "%j"))
doy_sc <- c()
for (i in 1:2) {
  doy_sc <- cbind(doy_sc, sin(2 * pi * i * doy / 365.25),
                  cos(2 * pi * i * doy / 365.25))
}

# Tabular predictors

x <- cbind(doy_sc, y_lag1)

##
# Image data (mean sea level pressure and sea level pressure gradient; 32 x 32)

nc <- ncdf4::nc_open("CCCRIS-181947_psl2_ERA5.nc")
psl <- ncdf4::ncvar_get(nc, varid = "psl")
ncdf4::nc_close(nc)

nc <- ncdf4::nc_open("CCCRIS-181947_psl-grad2_ERA5.nc")
psl_grad <- ncdf4::ncvar_get(nc, varid = "psl_grad")
ncdf4::nc_close(nc)

# Scale image predictors based on training split statistics
psl_mean <- apply(psl[, , custom_split], c(1, 2), mean)
psl_sd <- apply(psl[, , custom_split], c(1, 2), sd)
psl <- sweep(sweep(psl, c(1, 2), psl_mean, "-"), c(1, 2), psl_sd, "/")

psl_grad_mean <- apply(psl_grad[, , custom_split], c(1, 2), mean)
psl_grad_sd <- apply(psl_grad[, , custom_split], c(1, 2), sd)
psl_grad <- sweep(sweep(psl_grad, c(1, 2), psl_grad_mean, "-"), c(1, 2),
                  psl_grad_sd, "/")

x_image <- abind::abind(psl, psl_grad, along = -1)

# Reshape [time x channels x lon x lat]
x_image <- aperm(x_image, c(4, 1, 2, 3))
x_image <- x_image[-1, , , ]

##
# Number of mixtures and constraints informed by model-based clustering
# (mclust) on every 7th day (approximately independent events)

mc <- Mclust(y[which(custom_split)[c(TRUE, rep(FALSE, 6))], ], G = 1:10)
print(mc$BIC)

##
# Tabular module definition

tabular_module <- nn_module(
  "TabularModule",
  initialize = function(
    input_dim,
    hidden_dims,
    output_dim,
    dropout_rate
  ) {
    # Number of hidden layers
    if (is.null(hidden_dims) || length(hidden_dims) == 0) {
      # No hidden layers
      self$n_hidden_layers <- 0
      self$hidden_dims <- c()
    } else if (!is.vector(hidden_dims) && !is.list(hidden_dims)) {
      # Single hidden size passed, wrap into vector
      self$hidden_dims <- c(hidden_dims)
      self$n_hidden_layers <- length(self$hidden_dims)
    } else {
      # Vector or list of hidden sizes
      self$hidden_dims <- hidden_dims
      self$n_hidden_layers <- length(self$hidden_dims)
    }
    # Store output size and dropout rate
    self$output_dim <- output_dim
    self$dropout_rate <- dropout_rate
    # Module lists for linear layers, batch-norms, (optional) dropouts
    self$layers <- nn_module_list()
    self$bns <- nn_module_list()
    if (self$dropout_rate > 0) {
      self$dropouts <- nn_module_list()
    }
    # Build hidden layers
    current_dim <- input_dim
    if (self$n_hidden_layers > 0) {
      for (i in seq_len(self$n_hidden_layers)) {
        # Linear transform
        self$layers$append(
          nn_linear(current_dim, self$hidden_dims[[i]])
        )
        # Batch normalization on hidden size
        self$bns$append(
          nn_batch_norm1d(self$hidden_dims[[i]])
        )
        # Optional dropout after activation
        if (self$dropout_rate > 0) {
          self$dropouts$append(
            nn_dropout(p = self$dropout_rate)
          )
        }
        # Update input size for next layer
        current_dim <- self$hidden_dims[[i]]
      }
    }
    # Final linear layer: last hidden (or input) → output_dim
    self$layers$append(
      nn_linear(current_dim, output_dim)
    )
  },
  forward = function(x) {
    # Pass through each hidden layer
    for (i in seq_len(self$n_hidden_layers)) {
      x <- self$layers[[i]](x)  # linear
      x <- self$bns[[i]](x)     # batch-norm
      x <- nnf_relu(x)          # activation
      # Apply dropout if configured
      if (self$dropout_rate > 0 && !is.null(self$dropouts[[i]])) {
        x <- self$dropouts[[i]](x)
      }
    }
    # Final projection and activation
    x <- self$layers[[length(self$layers)]](x)
    x <- nnf_relu(x)
    x
  }
)

##
# Image module definition

image_module <- nn_module(
  "ImageModule",
  initialize = function(
    in_channels,
    img_size,
    conv_channels,
    kernel_size = 3,
    pool_kernel = 2,
    output_dim = 32
  ) {
    # Store output dim
    self$output_dim <- output_dim
    # Build conv stack
    self$n_conv <- length(conv_channels)
    self$convs <- nn_module_list()
    self$bn_conv <- nn_module_list()
    # Track spatial dim through conv+pool
    spatial <- img_size
    pad <- floor(kernel_size / 2)
    for (i in seq_along(conv_channels)) {
      in_ch <- if (i == 1) in_channels else conv_channels[i-1]
      out_ch <- conv_channels[i]
      # conv keeps spatial size (with padding)
      self$convs$append(
        nn_conv2d(
          in_channels = in_ch,
          out_channels = out_ch,
          kernel_size = kernel_size,
          padding = pad
        )
      )
      self$bn_conv$append(nn_batch_norm2d(out_ch))
      # Pooling halves spatial dims
      spatial <- floor(spatial / pool_kernel)
    }
    # Store pooling layer and computed flatten_dim
    self$pool <- nn_max_pool2d(kernel_size = pool_kernel)
    self$flatten_dim <- tail(conv_channels, 1) * spatial * spatial
    # Final head: linear( flatten_dim → output_dim ) + BN
    self$fc    <- nn_linear(self$flatten_dim, output_dim)
    self$bn_fc <- nn_batch_norm1d(output_dim)
  },
  forward = function(x) {
    # conv → BN → ReLU → pool
    for (i in seq_len(self$n_conv)) {
      x <- self$convs[[i]](x)
      x <- self$bn_conv[[i]](x)
      x <- nnf_relu(x)
      x <- self$pool(x)
    }
    # Flatten and head
    x <- torch_flatten(x, start_dim = 2)
    x <- self$fc(x)
    nnf_relu(self$bn_fc(x))
  }
)

##
# The TabularModule takes an input vector of length input_dim, runs it through 
# two dense layers (input_dim→32 and 32→16) each with batch-norm (BN), ReLU and
# 50 %  dropout, then applies a final 16→16 linear layer plus ReLU to produce a 
# 16-dimensional output.

tabular_mod <- tabular_module(
  input_dim = ncol(x),
  hidden_dims = c(32, 16),
  output_dim = 16,
  dropout_rate = 0.5
)

##
# The ImageModule accepts a 2×32×32 image, applies a 3×3 conv (2→16) with BN, 
#  ReLU  and 2×2 max-pool (→16×16), repeats with a 16→32 conv + BN, ReLU and 
# max-pool (→8×8), flattens the 32×8×8 tensor to 2048 units, and then projects
# it to 32 features via a linear layer, BN, and ReLU. Weight penalty (wd_image)
# is applied during training.

image_mod <- image_module(
  in_channels = dim(x_image)[2],
  img_size = dim(x_image)[3],
  conv_channels = c(16, 32),
  kernel_size = 3,
  pool_kernel = 2,
  output_dim = 32
)
wd_image <- 0.2

##
# Dense fusion network that processes concatenated tabular and image features
# and passes to the MST-PMDN head.

hidden_dim <- c(64, 32)
drop_hidden <- 0.5

## MST-MDN head
# Predicts parameters of the mixture of MST distributions based on outputs
# from the fusion network. Volume (L)-Shape (A)-Orientation (D) and MST
# constraints applied

modelname <- "VVI"
skewtname <- "FN"
constant_attr <- ""
n_mixtures <- 2
fixed_nu <- c(rep(50, n_mixtures - 1), NA)

cat(modelname, skewtname, constant_attr, n_mixtures, "\n")

out.pt <- paste0("wave-surge-dailymax.", modelname, skewtname,
                 constant_attr, n_mixtures, ".pt")
out.RData <- paste0("wave-surge-dailymax.", modelname, skewtname,
                    constant_attr, n_mixtures, ".RData")
out.pdf <- paste0("wave-surge-dailymax.", modelname, skewtname,
                  constant_attr, n_mixtures, ".pdf")

##
# Training/validation of deep MST-PMDN network

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
  nu_switch = 20.,
  max_alpha = 5.,
  min_vol_shape = 0.01,
  jitter = 1e-4,
  lr = 0.0001,
  max_norm = 1.,
  epochs = 200,
  batch_size = 32,
  drop_hidden = drop_hidden,
  wd_image = wd_image,
  wd_tabular = 0.,
  checkpoint_interval = 10,
  checkpoint_path = "wave-surge-checkpoint.pt",
  resume_from_checkpoint = FALSE,
  model = NULL,
  early_stopping_patience = 50,
  validation_split = 0,
  custom_split = custom_split,
  scheduler_step = 50,
  scheduler_gamma = 0.5,
  image_inputs = x_image,
  image_module = image_mod,
  tabular_module = tabular_mod,
  device = device
)

t2 <- Sys.time()
print(t2 - t1)

##
# Prediction using both tabular and image inputs.

pred <- predict_mst_pmdn(fit$model, x, x_image, device = device)
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

cat("Validation statistics:\n")
print(cor(y[!custom_split, ]))
print(cor(samples[!custom_split, ]))

print(cor(x[!custom_split, ], y[!custom_split, ]))
print(cor(y[!custom_split, ], samples[!custom_split, ]))

##
# Stochastic ensemble generation on validation split

x_valid <- x[!custom_split, ]
x_image_valid <- x_image[!custom_split, , , ]
y_valid <- y[!custom_split, ]

n_ens <- 30
rsamples_ens <- list()
for(iii in seq(n_ens)){
    rsamples_iii <- matrix(0, ncol = 2, nrow = nrow(x_valid))
    rsamples_iii[1, ] <- y_valid[1, ]
    cat(iii, '[', nrow(x_valid), '] : ')
    for (i in 2:nrow(x_valid)) {
      if (i %% 10 == 0) cat(i, "")
      x_valid_i <- cbind(x_valid[i, 1:(ncol(x) - 2), drop = FALSE],
                         rsamples_iii[i - 1, , drop = FALSE])
      x_image_valid_i <- x_image_valid[i , , , , drop=FALSE]
      rsamples_i <- sample_mst_pmdn(predict_mst_pmdn(fit$model,
                                    x_valid_i, x_image_valid_i,
                                    device = device),
                                    num_samples = 1)$samples[1, , ]
      rsamples_iii[i, ] <- as.matrix(rsamples_i)
    }
    rsamples_iii[rsamples_iii[, 1] < y1_min, 1] <- y1_min
    colnames(rsamples_iii) <- paste("MST-PMDN", colnames(y_valid))
    rsamples_ens[[iii]] <- rsamples_iii
    cat("\n")
}

escore_valid <- y_valid[, 1] * NA
for(i in seq(nrow(y_valid))) {
  y_i <- y_valid[i, ]
  dat_i <- sapply(rsamples_ens, function(x, i) x[i, ], i = i)
  escore_valid[i] <- scoringRules::es_sample(y_i, dat_i)
}
cat("Energy score valid:", mean(escore_valid), "\n")

##
# Evaluation plots

pdf(out.pdf, width = 10, height = 8)

##
# Training and validation curves

matplot(cbind(fit$train_loss_history/fit$train_loss_history[1],
        fit$val_loss_history/fit$val_loss_history[1]), type = "b",
        pch=15, ylab = "Loss", xlab = "Epoch")
grid()
legend("topright", c("Train", "Validation"), col = c(1, 2), pch = 15)

dev.next()
par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))
plot(y_valid[, 1], y_valid[, 2], main = "Original Data",
     col = scales::alpha("black", 0.2), pch = 19, cex = 1.5,
     xlab = colnames(y)[1], ylab = colnames(y)[2])
plot(rsamples_ens[[1]][, 1], rsamples_ens[[1]][, 2], xlab = colnames(y)[1],
     ylab = colnames(y)[2], main = "MST-PMDN samples",
     col = scales::alpha("darkblue", 0.2), pch = 19, cex = 1.5)
image(MASS::kde2d(y_valid[, 1], y_valid[, 2]))
image(MASS::kde2d(rsamples_ens[[1]][, 1], rsamples_ens[[1]][, 2]))

dev.next()
par(mfrow = c(2, 2), mar = c(4, 4, 4, 1))
acf(y_valid[, 1, drop=FALSE], lwd = 2)
acf(y_valid[, 2, drop=FALSE], lwd = 2)
acf(rsamples_ens[[1]][, 1, drop=FALSE], col = "blue", lwd = 2)
acf(rsamples_ens[[1]][, 2, drop=FALSE], col = "blue", lwd = 2)

dev.next()
lims <- apply(do.call(rbind, rsamples_ens), 2, function(x) range(pretty(x)))
par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))
for(i in seq(ncol(y_valid))) {
    qqplot(y_valid[, i], y_valid[, i], main = colnames(y)[i],
         col = NA, pch = NA, xlab = "Original Data",
         ylab = "MST-PMDN samples", xlim = lims[, i], ylim = lims[, i])
    for(ens in seq_along(rsamples_ens)) {
        points(sort(y_valid[, i]), sort(rsamples_ens[[ens]][, i]),
               pch='+', col = scales::alpha("darkblue", 0.05))
    }
    grid()
    abline(0, 1, lty = 2)
}
for(i in seq(ncol(y_valid))) {
    plot(as.Date(date[!custom_split]), y_valid[, i], pch = 1, cex = 0.75,
         xlab = "Date", ylab = colnames(y)[i], ylim = lims[, i])
    for(ens in seq_along(rsamples_ens)) {
        points(as.Date(date[!custom_split]), rsamples_ens[[ens]][, i],
               pch='+', col = scales::alpha("darkblue", 0.05))
    }
    grid()
}

##
# Time series of degrees-of-freedom nu

dev.next()
plot(as.Date(date[!custom_split]), pred$nu[!custom_split, n_mixtures],
     xlab = "Year", ylab = expression(nu), ylim=c(0, 51), type = "p",
     pch = 15, col = scales::alpha("blue", 0.5))
grid()
abline(h = 3, col = "red")
abline(h = 50, col = "black")
abline(h = 30, col = "dark blue", lty = 2)

##
# Ensemble verification

multivar_rank_histograms_4panel <- function(
    obs, ens,
    main_titles = c("Energy Score", "Mean", "Variance", "Half-space Depth"),
    xlab = "Rank", ylab = "Frequency",
    nbins = NULL, plot = TRUE,
    plot_density = TRUE
) {
  if (!requireNamespace("ddalpha", quietly = TRUE)) {
    stop("Package 'ddalpha' is required for half-space depth.", call. = FALSE)
  }
  N <- nrow(obs)
  d <- ncol(obs)
  M <- length(ens)
  # Validate input dimensions
  if (!all(sapply(ens, function(e) is.matrix(e) && all(dim(e) == c(N, d))))) {
    stop("Each element of 'ens' must have dim the same as 'obs' (N x d).")
  }
  if (d < 1) stop("'obs' and 'ens' members must have d >= 1.")
  # Pre-rank functions
  energy_score <- function(x0_vec, Xm_mat) {
    current_K <- nrow(Xm_mat) 
    if (is.null(current_K) || current_K == 0) return(NA)
    term1 <- mean(sqrt(rowSums(sweep(Xm_mat, 2, x0_vec, "-")^2)))
    if (current_K == 1) { 
        term2 <- 0 
    } else {
        term2 <- 0.5 * mean(dist(Xm_mat))
    }
    return(term1 - term2)
  }
  mean_prerank <- function(x_vec) mean(x_vec)
  variance_prerank <- function(x_vec) {
    if (length(x_vec) == 0) return(NA)
    m_x <- mean(x_vec)
    return(mean((x_vec - m_x)^2))
  }
  rank_lists <- list(
    energy = integer(N),
    mean = integer(N),
    variance = integer(N),
    halfspace_depth = integer(N)
  )
  for (i in 1:N) {
    Xm_list <- lapply(ens, function(e) e[i, ])
    if (d == 1) {
        Xm <- matrix(unlist(Xm_list), ncol = 1)
    } else {
        Xm <- do.call(rbind, Xm_list)
    }
    
    if(nrow(Xm) != M || ncol(Xm) != d) {
        stop(paste0("Xm dimensions are incorrect at iteration i=", i, 
                   ". Expected ", M, "x", d, ", Got ", nrow(Xm), "x", ncol(Xm)))
    }
    obs_i_vec <- obs[i, ]
    obs_i_mat <- matrix(obs_i_vec, nrow = 1, ncol = d)
    X_all_i <- rbind(Xm, obs_i_mat)
    # Energy Score Rank (Symmetrical)
    all_es_values <- sapply(1:(M + 1), function(k) {
      energy_score(x0_vec = X_all_i[k, ], Xm_mat = X_all_i)
    })
    if(anyNA(all_es_values)) { rank_lists$energy[i] <- NA
    } else { rank_lists$energy[i] <- rank(all_es_values, ties.method = "random",
      na.last = "keep")[M + 1] }
    # Mean Rank
    mean_ens <- rowMeans(Xm)
    mean_obs <- mean(obs_i_vec)
    all_mean <- c(mean_ens, mean_obs)
    if(anyNA(all_mean)) { rank_lists$mean[i] <- NA
    } else { rank_lists$mean[i] <- rank(all_mean, ties.method = "random",
      na.last = "keep")[M + 1] }
    # Variance Rank
    var_ens <- apply(Xm, 1, variance_prerank)
    var_obs <- variance_prerank(obs_i_vec)
    all_var <- c(var_ens, var_obs)
     if(anyNA(all_var)) { rank_lists$variance[i] <- NA
    } else { rank_lists$variance[i] <- rank(all_var, ties.method = "random",
      na.last = "keep")[M + 1] }
    # Half-space Depth Rank (Symmetrical)
    all_hs_depth_values <- ddalpha::depth.halfspace(x = X_all_i, data = X_all_i)
    if(anyNA(all_hs_depth_values)) { 
      rank_lists$halfspace_depth[i] <- NA
    } else {
      rank_lists$halfspace_depth[i] <- rank(all_hs_depth_values,
      ties.method = "random", na.last = "keep")[M + 1]
    }
  }
  if (is.null(nbins)) nbins <- M + 1
  # `bin_edges` define the boundaries for hist. Ranks are integers from 1 to M+1.
  # These edges ensure that each integer rank (if nbins = M+1) falls in the
  # middle of a bin.
  bin_edges <- seq(0.5, M + 1.5, length.out = nbins + 1) 
  # Determine bar_names for x-axis labels
  if (nbins == M + 1) {
    bar_names_to_use <- as.character(1:(M + 1))
  } else {
    bar_names_to_use <- character(nbins)
    for (j in 1:nbins) {
      r_start <- if (j == 1) { ceiling(bin_edges[j]) } else {
        floor(bin_edges[j]) + 1 }
      r_end <- floor(bin_edges[j+1])
      r_start <- max(1, r_start); r_end <- min(M + 1, r_end)
      if (r_start == r_end) { bar_names_to_use[j] <- as.character(r_start)
      } else if (r_start > r_end) { mid_val <- (bin_edges[j] + 
        bin_edges[j+1]) / 2
        bar_names_to_use[j] <- sprintf("%.0f", mid_val)
      } else { bar_names_to_use[j] <- paste0(r_start, "-", r_end) }
    }
  }
  numeric_bin_centers <- (bin_edges[-1] + bin_edges[-length(bin_edges)]) / 2
  if (plot) {
    op <- par(mfrow = c(2, 2), mar = c(4.5, 4.5, 2.1, 1), oma = c(0,0,2,0))
    plot_types <- c("energy", "mean", "variance", "halfspace_depth")
    current_ylab <- if(plot_density) "Density" else ylab
    for (k_idx in 1:length(plot_types)) {
      k <- plot_types[k_idx]
      current_ranks <- rank_lists[[k]]
      # Determine reference line height and y-limits based on count or density
      if (plot_density) {
        reference_line_h <- 1 / (M + 1) 
      } else {
        reference_line_h <- N / nbins
      }
      if(all(is.na(current_ranks))) {
          plot(1, type="n", xlab=xlab, ylab=current_ylab,
               main=main_titles[k_idx],  xlim=c(0.5, M+1.5), 
               ylim=if(plot_density) c(0,
               reference_line_h * 2 + 0.01) else c(0, N))
          text(mean(par("usr")[1:2]), mean(par("usr")[3:4]),
          "All rank data are NA", cex=1.2)
          box()
          abline(h = reference_line_h, col = "red", lty = 2, lwd=1.5); next
      }
      # Calculate histogram data: either counts or densities
      h <- hist(current_ranks, breaks = bin_edges, plot = FALSE,
      probability = plot_density)
      height_values <- if(plot_density) h$density else h$counts
      # Adjust ylim dynamically
      if (plot_density) {
          ylim_val <- c(0, max(reference_line_h * 2,
          max(height_values, na.rm = TRUE) * 1.1, na.rm = TRUE) + 0.01)
      } else {
          ylim_val <- c(0, max(reference_line_h * 1.5,
          max(height_values, na.rm = TRUE) * 1.1, na.rm = TRUE) + 1)
      }
      if(ylim_val[2] <= ylim_val[1]) ylim_val[2] <- ylim_val[1] + 1
      barplot(height_values, names.arg = bar_names_to_use,
              main = main_titles[k_idx], xlab = xlab, ylab = current_ylab,
              col = "#1E90FFB3", border = "white",  ylim = ylim_val,
              cex.names = 0.8, cex.axis = 0.8, cex.lab = 0.9, cex.main = 1)
      abline(h = reference_line_h, col = "red", lty = 2, lwd=1.5); box()
    }
    par(op)
  }
  invisible(list(
    ranks = rank_lists,
    bin_edges = bin_edges,
    bin_centers = numeric_bin_centers,
    is_density_plot = plot_density
  ))
}

dev.next()
mrh_valid <- multivar_rank_histograms_4panel(y_valid, rsamples_ens,
               nbins = n_ens+1)

dev.off()

##
# Save pt and RData files

torch_save(fit$model, out.pt)
save.image(file = out.RData)

################################################################################
