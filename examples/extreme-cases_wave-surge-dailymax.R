library(MST.PMDN)
library(MASS)
library(ggplot2)
library(dplyr)

# Load model and test data
load("wave-surge-dailymax.VVIFN2.RData")
fit$model <- torch_load("wave-surge-dailymax.VVIFN2.pt")
pred_test <- predict_mst_pmdn(fit$model, x_test, x_image_test)
pred_test_pi <- cbind(1:nrow(y_test), as.array(pred_test$pi))

# Simulation draws/KDE and coverage levels for nested contours
probs <- c(0.05, 0.1, 0.25, 0.5, 0.75)
n_draws <- 1000000
n_grid <- 100

# Identify "extreme" days
use_bimodal <- TRUE
thresh_l <- 2.5
thresh_u <- Inf
is_extreme   <- ((abs(y_test[,1]) > thresh_l) | (abs(y_test[,2]) > thresh_l)) &
                ((abs(y_test[,1]) < thresh_u) | (abs(y_test[,2]) < thresh_u))
idx_extreme  <- which(is_extreme)

idx_bimodal <- which.max(apply(y_test, 1, max) * 
  (apply(pred_test_pi[,-1], 1, max) < 0.6))
if (use_bimodal) idx_extreme <- c(idx_extreme, idx_bimodal)

# Prepare containers
obs_list     <- list()
contour_list <- list()
sample_list  <- list()

for (j in seq_along(idx_extreme)) {
  i      <- idx_extreme[j]
  day_id <- gsub("-", "", date[!custom_split][i])

  # 1) Single-day predictors
  x_i       <- matrix( x_test[i, , drop = FALSE], nrow = 1 )
  x_image_i <- array(x_image_test[i, , , , drop = FALSE], dim = c(1,2,32,32))

  # 2) Predict MST-PMDN parameters
  pred_i <- predict_mst_pmdn(fit$model, x_i, x_image_i, device = "cpu")
  print(pred_i$pi)

  # 3) Mixture mean
  pi_vec <- as.numeric(pred_i$pi)          # length M
  mu_arr <- as_array(pred_i$mu)            # [1 × M × 2]
  mu_mat <- matrix(mu_arr[1,,], ncol = 2)  # [M × 2]
  mix_mean <- colSums(pi_vec * mu_mat)     # length 2

  # 4) Draw ensemble
  sample_tensor <- sample_mst_pmdn(pred_i, num_samples = n_draws, device = "cpu")
  samp_mat     <- as.matrix(torch_squeeze(sample_tensor$samples, dim = 2))
  colnames(samp_mat) <- c("wave","surge")

  # 5) KDE + nested contours
  kd    <- kde2d(samp_mat[,"wave"], samp_mat[,"surge"], n = n_grid)
  dx    <- diff(kd$x[1:2]); dy <- diff(kd$y[1:2])
  zz    <- sort(as.vector(kd$z), decreasing = TRUE)
  cumz  <- cumsum(zz) * dx * dy

  # density thresholds for each coverage
  level_z <- vapply(probs, function(u) {
    zz[ which(cumz >= u)[1] ]
  }, numeric(1))

  # extract all contours, assign seg_id to each disjoint loop
  cls_all <- contourLines(x = kd$x, y = kd$y, z = kd$z, levels = level_z)
  df_cnts <- lapply(seq_along(cls_all), function(k) {
    seg <- cls_all[[k]]
    data.frame(
      day_id = day_id,
      prob   = probs[ match(seg$level, level_z) ],
      seg_id = k,                # new segment identifier
      wave   = seg$x,
      surge  = seg$y
    )
  }) %>% bind_rows()
  contour_list[[j]] <- df_cnts

  # 6) Record observed and mean
  obs_list[[j]] <- data.frame(
    day_id     = day_id,
    wave_obs   = y_test[i,1],
    surge_obs  = y_test[i,2],
    wave_mean  = mix_mean[1],
    surge_mean = mix_mean[2]
  )

  # (Optional) record samples
  sample_list[[j]] <- data.frame(
    day_id = rep(day_id, n_draws),
    wave   = samp_mat[,"wave"],
    surge  = samp_mat[,"surge"]
  )
}

# Combine results
obs_df     <- bind_rows(obs_list)
contour_df <- bind_rows(contour_list)
sample_df  <- bind_rows(sample_list)

# All observed test points
obs_all_df <- data.frame(
  wave  = y_test[,1],
  surge = y_test[,2]
)

p <- ggplot() +
  # base layer: all observed points
  geom_point(
    data = obs_all_df,
    aes(x = wave, y = surge),
    color = "grey70", shape = 16, size = 5, alpha = 0.5,
    show.legend = FALSE
  ) +
  # nested contours, grouping also by seg_id
  geom_path(
    data = contour_df,
    aes(
      x        = wave,
      y        = surge,
      group    = interaction(day_id, prob, seg_id),  # include seg_id
      color    = day_id,
      linetype = factor(prob)
    ),
    linewidth = 1.0, alpha = 0.5, show.legend = FALSE
  ) +
  # observed extremes
  geom_point(
    data = obs_df,
    aes(x = wave_obs, y = surge_obs, color = day_id),
    shape = 18, size = 3, stroke = 1.2
  ) +
  # mixture means
  geom_point(
    data = obs_df,
    aes(x = wave_mean, y = surge_mean, fill = day_id),
    shape = 21, size = 3, stroke = 1.2, color = "black"
  ) +
  # linking segments
  geom_segment(
    data = obs_df,
    aes(
      x = wave_obs, y = surge_obs,
      xend = wave_mean, yend = surge_mean,
      color = day_id
    ),
    linetype = "dashed", linewidth = 0.8
  ) +
  scale_color_brewer(palette = "Set1") +
  scale_fill_brewer(palette = "Set1") +
  scale_linetype_manual(
    name = "Coverage",
    values = c(
      "0.75" = "solid",
      "0.5"  = "dashed",
      "0.25" = "solid",
      "0.1"  = "dashed",
      "0.05" = "solid"
    )
  ) +
  labs(
    x = "Daily max wave (standard deviations)",
    y = "Daily max surge (standard deviations)",
    title = ""
  ) +
  geom_hline(yintercept = 0, color = "black", linewidth = 1, linetype = "dashed") +
  geom_vline(xintercept = 0, color = "black", linewidth = 1, linetype = "dashed") +
  theme_minimal() +
  theme(
    legend.position     = "bottom",
    legend.title        = element_text(size = 10),
    panel.border        = element_rect(color = "black", fill = NA, linewidth = 1),
    panel.background    = element_blank(),
    axis.ticks.length   = unit(5, "pt"),
    axis.ticks.x.bottom = element_line(color = "black"),
    axis.ticks.y.left   = element_line(color = "black"),
    axis.text           = element_text(size = 16),
    axis.title          = element_text(size = 18)
  )

x11()
print(p)