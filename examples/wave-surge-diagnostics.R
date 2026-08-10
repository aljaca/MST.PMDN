################################################################################
# Diagnostics for the revised MST.PMDN wave-surge example
#
# Usage:
#   Rscript wave-surge-diagnostics.R
#   Rscript wave-surge-diagnostics.R results.rds diagnostics.pdf
#   Rscript wave-surge-diagnostics.R results.rds diagnostics.pdf wave_surge.RData
#
# The compact results RDS supplies fitted outputs and optimization histories.
# Response observations, transformations, and normalization statistics come
# from the frozen wave_surge preprocessing object. The neural network is not
# loaded or refitted.
################################################################################

args <- commandArgs(trailingOnly = TRUE)
results_file <- if (length(args) >= 1L) {
  args[1]
} else {
  "out/wave-surge-dailymax-results.rds"
}
figure_file <- if (length(args) >= 2L) {
  args[2]
} else if (grepl("-results[.]rds$", results_file)) {
  sub("-results[.]rds$", "-diagnostics.pdf", results_file)
} else {
  paste0(tools::file_path_sans_ext(results_file), "-diagnostics.pdf")
}
data_file <- if (length(args) >= 3L) {
  args[3]
} else {
  "wave_surge.RData"
}

if (!file.exists(results_file)) {
  stop("Results file not found: ", results_file)
}
if (!file.exists(data_file)) {
  stop("Preprocessed data file not found: ", data_file)
}
for (pkg in c("ddalpha", "MASS", "scoringRules")) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop("Package '", pkg, "' is required.")
  }
}

results <- readRDS(results_file)
required_results <- c(
  "test_dates", "recursive_samples", "train_loss_history", "val_loss_history"
)
missing_items <- setdiff(required_results, names(results))
if (length(missing_items) > 0L) {
  stop(
    "The results file is missing: ",
    paste(missing_items, collapse = ", ")
  )
}

data_environment <- new.env(parent = baseenv())
loaded_objects <- load(data_file, envir = data_environment)
if (!identical(loaded_objects, "wave_surge") ||
    !is.list(data_environment$wave_surge)) {
  stop(data_file, " must contain one list named 'wave_surge'.")
}
wave_surge <- data_environment$wave_surge
rm(data_environment)

required_data <- c(
  "date", "y", "y_original", "y_mean", "y_sd", "wave_scale",
  "inv_softplus", "softplus"
)
missing_items <- setdiff(required_data, names(wave_surge))
if (length(missing_items) > 0L) {
  stop(
    "The wave_surge object is missing: ",
    paste(missing_items, collapse = ", ")
  )
}

data_dates <- as.Date(wave_surge$date)
data_y <- as.matrix(wave_surge$y)
data_y_original <- as.matrix(wave_surge$y_original)
wave_scale <- as.numeric(wave_surge$wave_scale)
target_center <- as.numeric(wave_surge$y_mean)
target_scale <- as.numeric(wave_surge$y_sd)
inv_softplus <- wave_surge$inv_softplus
softplus <- wave_surge$softplus

# The saved functions use only base-R operations. Detaching their environments
# prevents them from retaining the large x_image field after wave_surge is
# removed.
if (is.function(inv_softplus)) {
  environment(inv_softplus) <- baseenv()
}
if (is.function(softplus)) {
  environment(softplus) <- baseenv()
}
rm(wave_surge, loaded_objects)
invisible(gc())

dates <- as.Date(results$test_dates)
if (anyNA(dates) || anyNA(data_dates) || anyDuplicated(dates) ||
    anyDuplicated(data_dates)) {
  stop("Result and preprocessing dates must be valid and unique.")
}
test_rows <- match(dates, data_dates)
if (anyNA(test_rows)) {
  stop("Every test date must occur in wave_surge$date.")
}

observed <- data_y_original[test_rows, , drop = FALSE]
observed_standard <- data_y[test_rows, , drop = FALSE]
rm(data_y, data_y_original)
samples <- results$recursive_samples
if (length(dim(samples)) != 3L ||
    dim(samples)[1] != nrow(observed) ||
    dim(samples)[3] != ncol(observed) ||
    length(dates) != nrow(observed)) {
  stop(
    "Expected recursive_samples in [date, member, response] order with ",
    "one case per matched wave_surge test date."
  )
}
if (ncol(observed) != 2L) {
  stop("These diagnostics expect exactly two responses: wave height and surge.")
}
if (any(!is.finite(observed)) || any(!is.finite(samples))) {
  stop("Observed values and recursive samples must all be finite.")
}

n_test <- nrow(observed)
n_ens <- dim(samples)[2]
n_response <- ncol(observed)
response_names <- colnames(observed)
if (is.null(response_names)) {
  response_names <- c("Wave.m", "Surge.m")
}
dimnames(samples)[[3]] <- response_names

if (length(wave_scale) != 1L || !is.finite(wave_scale) || wave_scale <= 0 ||
    length(target_center) != n_response ||
    length(target_scale) != n_response ||
    any(!is.finite(target_center)) ||
    any(!is.finite(target_scale)) ||
    any(target_scale <= 0) ||
    !is.function(inv_softplus) || !is.function(softplus)) {
  stop("Invalid response transformations or normalization statistics.")
}

# Transform arrays whose final dimension contains wave height then surge.
to_standardized_response <- function(values) {
  value_dim <- dim(values)
  value_dimnames <- dimnames(values)
  transformed <- matrix(values, ncol = n_response)
  transformed[, 1] <- inv_softplus(transformed[, 1], wave_scale)
  standardized <- sweep(transformed, 2, target_center, "-")
  standardized <- sweep(standardized, 2, target_scale, "/")
  dim(standardized) <- value_dim
  dimnames(standardized) <- value_dimnames
  standardized
}

from_standardized_response <- function(values) {
  value_dim <- dim(values)
  value_dimnames <- dimnames(values)
  transformed <- matrix(values, ncol = n_response)
  transformed <- sweep(transformed, 2, target_scale, "*")
  transformed <- sweep(transformed, 2, target_center, "+")
  transformed[, 1] <- softplus(transformed[, 1], wave_scale)
  dim(transformed) <- value_dim
  dimnames(transformed) <- value_dimnames
  transformed
}

observed_standard_check <- to_standardized_response(observed)
observed_round_trip <- from_standardized_response(observed_standard)
if (!isTRUE(all.equal(
  observed_standard_check, observed_standard,
  tolerance = 1e-10, check.attributes = FALSE
)) || !isTRUE(all.equal(
  observed_round_trip, observed,
  tolerance = 1e-10, check.attributes = FALSE
))) {
  stop("Saved response data and transformations are inconsistent.")
}
rm(observed_standard_check, observed_round_trip)
if (!is.null(results$observed_test) && !isTRUE(all.equal(
  as.matrix(results$observed_test), observed,
  tolerance = 1e-12, check.attributes = FALSE
))) {
  stop("Results and wave_surge.RData contain different test observations.")
}

# Put wave height and surge on the standardized model scale before computing
# multivariate Euclidean distances, means, or variances. This avoids allowing
# either response to dominate solely because of its physical units.
samples_standard <- to_standardized_response(samples)
if (any(!is.finite(observed_standard)) ||
    any(!is.finite(samples_standard))) {
  stop("Response transformation produced non-finite standardized values.")
}

diagnostic_seed <- if (is.null(results$seed)) {
  1747892421L
} else {
  as.integer(results$seed)
}
set.seed(diagnostic_seed + 1L)

random_rank <- function(values, target) {
  rank(values, ties.method = "random", na.last = "keep")[target]
}

fan_plot <- function(
    x, y, ensemble, ylab, main, colour,
    probs = c(0.05, 0.25, 0.5, 0.75, 0.95)
) {
  q <- t(apply(
    ensemble, 1, quantile, probs = probs, type = 8, names = FALSE
  ))
  ylim <- range(c(y, q), finite = TRUE)
  plot(
    x, y, type = "n", xlab = "Date", ylab = ylab,
    main = main, ylim = ylim
  )
  polygon(
    c(x, rev(x)), c(q[, 1], rev(q[, 5])),
    border = NA, col = grDevices::adjustcolor(colour, alpha.f = 0.16)
  )
  polygon(
    c(x, rev(x)), c(q[, 2], rev(q[, 4])),
    border = NA, col = grDevices::adjustcolor(colour, alpha.f = 0.28)
  )
  lines(x, q[, 3], col = colour, lwd = 1.5)
  lines(x, y, col = "black", lwd = 1)
  grid()
  legend(
    "topleft",
    legend = c("Observed", "Ensemble median", "50% / 90% intervals"),
    col = c("black", colour, grDevices::adjustcolor(colour, alpha.f = 0.35)),
    lty = c(1, 1, NA), lwd = c(1, 1.5, NA),
    pch = c(NA, NA, 15), pt.cex = 1.5, bty = "n", cex = 0.8
  )
}

# Pool the M + 1 possible ranks because 179 dates cannot support a readable
# 51-bin histogram. Reference intervals assume independent forecast cases and
# are therefore descriptive for this serially dependent winter trajectory.
rank_bins <- min(10L, n_ens + 1L)
rank_to_bin <- cut(
  seq_len(n_ens + 1L),
  breaks = seq(0.5, n_ens + 1.5, length.out = rank_bins + 1L),
  labels = FALSE, include.lowest = TRUE
)
rank_bin_probability <- tabulate(
  rank_to_bin, nbins = rank_bins
) / (n_ens + 1L)
rank_bin_labels <- vapply(seq_len(rank_bins), function(k) {
  values <- which(rank_to_bin == k)
  if (length(values) == 1L) {
    as.character(values)
  } else {
    paste0(min(values), "-", max(values))
  }
}, character(1))

pooled_rank_plot <- function(ranks, main, colour) {
  rank_bin <- rank_to_bin[ranks]
  counts <- tabulate(rank_bin, nbins = rank_bins)
  expected <- length(ranks) * rank_bin_probability
  lower <- stats::qbinom(0.025, length(ranks), rank_bin_probability)
  upper <- stats::qbinom(0.975, length(ranks), rank_bin_probability)
  ylim <- c(0, max(c(counts, upper)) * 1.12)
  at <- barplot(
    counts, names.arg = rank_bin_labels, col = colour, border = "white",
    xlab = "Rank range", ylab = "Count", main = main, ylim = ylim,
    cex.names = 0.75
  )
  arrows(
    at, lower, at, upper, angle = 90, code = 3, length = 0.035,
    col = "grey30", lwd = 1
  )
  points(at, expected, pch = 4, col = "#D55E00", lwd = 2)
  mtext(
    "x: expected; whiskers: pointwise 95% independence reference",
    side = 3, line = 0.15, adj = 1, cex = 0.55, col = "grey30"
  )
  box()
}

# Marginal observation ranks.
marginal_ranks <- matrix(
  NA_integer_, nrow = n_test, ncol = n_response,
  dimnames = list(NULL, response_names)
)
for (i in seq_len(n_test)) {
  for (j in seq_len(n_response)) {
    marginal_ranks[i, j] <- random_rank(
      c(samples[i, , j], observed[i, j]),
      n_ens + 1L
    )
  }
}

# Multivariate ranks. The energy-score calculation treats each of the M + 1
# vectors in turn as the outcome and the remaining M vectors as its ensemble.
# This leave-one-out construction is permutation-symmetric.
multivariate_ranks <- matrix(
  NA_integer_, nrow = n_test, ncol = 4L,
  dimnames = list(
    NULL,
    c("Energy score", "Location (mean)", "Scale (variance)",
      "Half-space depth")
  )
)
for (i in seq_len(n_test)) {
  values <- rbind(
    samples_standard[i, , ],
    observed_standard[i, ]
  )
  distance_matrix <- as.matrix(stats::dist(values))
  distance_sum <- rowSums(distance_matrix)
  total_distance <- sum(distance_matrix)
  energy_prerank <- distance_sum / n_ens -
    0.5 * (total_distance - 2 * distance_sum) / n_ens^2
  location_prerank <- rowMeans(values)
  scale_prerank <- rowMeans(
    sweep(values, 1, rowMeans(values), "-")^2
  )
  depth_prerank <- ddalpha::depth.halfspace(
    x = values, data = values
  )

  multivariate_ranks[i, ] <- c(
    random_rank(energy_prerank, n_ens + 1L),
    random_rank(location_prerank, n_ens + 1L),
    random_rank(scale_prerank, n_ens + 1L),
    random_rank(depth_prerank, n_ens + 1L)
  )
}

# The scale-balanced energy score is used in multivariate diagnostics. The
# physical-scale version is also retained so it can be reconciled with the
# score reported by wave-surge-dailymax.R.
energy_score_standardized <- vapply(seq_len(n_test), function(i) {
  scoringRules::es_sample(
    y = observed_standard[i, ],
    dat = t(samples_standard[i, , ])
  )
}, numeric(1))
energy_score_physical <- vapply(seq_len(n_test), function(i) {
  scoringRules::es_sample(
    y = observed[i, ],
    dat = t(samples[i, , ])
  )
}, numeric(1))
if (!is.null(results$energy_score) && !isTRUE(all.equal(
  as.numeric(results$energy_score), energy_score_physical,
  tolerance = 1e-10, check.attributes = FALSE
))) {
  stop("The saved and recomputed physical-scale energy scores disagree.")
}

crps <- matrix(
  NA_real_, nrow = n_test, ncol = n_response,
  dimnames = list(NULL, response_names)
)
for (i in seq_len(n_test)) {
  for (j in seq_len(n_response)) {
    crps[i, j] <- scoringRules::crps_sample(
      y = observed[i, j],
      dat = samples[i, , j]
    )
  }
}

# With two responses there is one unique variogram pair. p = 0.5 is less
# dominated by large response differences than the squared-increment choice.
variogram_p <- 0.5
observed_increment <- abs(
  observed_standard[, 1] - observed_standard[, 2]
)^variogram_p
forecast_increment <- rowMeans(abs(
  samples_standard[, , 1] - samples_standard[, , 2]
)^variogram_p)
variogram_score <- (observed_increment - forecast_increment)^2

ensemble_mean_standard <- apply(samples_standard, c(1, 3), mean)
ensemble_spread_standard <- apply(samples_standard, c(1, 3), sd)

# Central-interval coverage and quantile calibration.
coverage_levels <- c(0.50, 0.80, 0.90)
coverage <- matrix(
  NA_real_, nrow = length(coverage_levels), ncol = n_response,
  dimnames = list(coverage_levels, response_names)
)
for (j in seq_len(n_response)) {
  for (k in seq_along(coverage_levels)) {
    alpha <- (1 - coverage_levels[k]) / 2
    interval <- t(apply(
      samples[, , j], 1, quantile,
      probs = c(alpha, 1 - alpha), type = 8, names = FALSE
    ))
    coverage[k, j] <- mean(
      observed[, j] >= interval[, 1] &
        observed[, j] <= interval[, 2]
    )
  }
}

probability_levels <- seq(0.02, 0.98, by = 0.02)
quantile_calibration <- matrix(
  NA_real_, nrow = length(probability_levels), ncol = n_response,
  dimnames = list(probability_levels, response_names)
)
for (j in seq_len(n_response)) {
  forecast_quantile <- sapply(probability_levels, function(probability) {
    apply(
      samples[, , j], 1, quantile,
      probs = probability, type = 8, names = FALSE
    )
  })
  quantile_calibration[, j] <- colMeans(
    observed[, j] <= forecast_quantile
  )
}

# Dependence and temporal summaries retain complete recursive trajectories.
ensemble_correlation <- vapply(seq_len(n_ens), function(member) {
  cor(samples[, member, 1], samples[, member, 2])
}, numeric(1))
observed_correlation <- cor(observed[, 1], observed[, 2])

rolling_window <- min(30L, floor(n_test / 3))
rolling_index <- rolling_window:n_test
rolling_observed_correlation <- vapply(rolling_index, function(i) {
  rows <- (i - rolling_window + 1L):i
  cor(observed[rows, 1], observed[rows, 2])
}, numeric(1))
rolling_ensemble_correlation <- sapply(seq_len(n_ens), function(member) {
  vapply(rolling_index, function(i) {
    rows <- (i - rolling_window + 1L):i
    cor(samples[rows, member, 1], samples[rows, member, 2])
  }, numeric(1))
})
rolling_correlation_quantiles <- t(apply(
  rolling_ensemble_correlation, 1, quantile,
  probs = c(0.05, 0.5, 0.95), type = 8, names = FALSE
))

lag_max <- min(14L, floor(n_test / 4))
acf_observed <- sapply(seq_len(n_response), function(j) {
  as.numeric(stats::acf(
    observed[, j], lag.max = lag_max, plot = FALSE
  )$acf)[-1]
})
acf_ensemble <- array(
  NA_real_, dim = c(lag_max, n_ens, n_response)
)
for (member in seq_len(n_ens)) {
  for (j in seq_len(n_response)) {
    acf_ensemble[, member, j] <- as.numeric(stats::acf(
      samples[, member, j], lag.max = lag_max, plot = FALSE
    )$acf)[-1]
  }
}

lagged_correlation <- function(x, y, lags) {
  n <- length(x)
  vapply(lags, function(k) {
    if (k >= 0L) {
      cor(x[seq_len(n - k)], y[(1L + k):n])
    } else {
      h <- -k
      cor(x[(1L + h):n], y[seq_len(n - h)])
    }
  }, numeric(1))
}

cross_lags <- -7:7
cross_observed <- lagged_correlation(
  observed[, 1], observed[, 2], cross_lags
)
cross_ensemble <- sapply(seq_len(n_ens), function(member) {
  lagged_correlation(
    samples[, member, 1], samples[, member, 2], cross_lags
  )
})

# Equal-count spread groups on the standardized scale.
n_spread_groups <- 5L
spread_skill <- lapply(seq_len(n_response), function(j) {
  group <- ceiling(
    rank(ensemble_spread_standard[, j], ties.method = "first") /
      n_test * n_spread_groups
  )
  cbind(
    spread = tapply(
      ensemble_spread_standard[, j], group, mean
    ),
    rmse = sqrt(tapply(
      (ensemble_mean_standard[, j] - observed_standard[, j])^2,
      group, mean
    ))
  )
})

# Pooled joint samples for display only. Highest-density-region contour levels
# are estimated on a common physical response grid.
pooled_samples <- do.call(rbind, lapply(seq_len(n_ens), function(member) {
  samples[, member, ]
}))
joint_limits <- c(
  range(c(observed[, 1], pooled_samples[, 1])),
  range(c(observed[, 2], pooled_samples[, 2]))
)
observed_density <- MASS::kde2d(
  observed[, 1], observed[, 2], n = 80, lims = joint_limits
)
forecast_density <- MASS::kde2d(
  pooled_samples[, 1], pooled_samples[, 2], n = 80, lims = joint_limits
)
hdr_levels <- function(density, probabilities = c(0.95, 0.80, 0.50)) {
  z <- sort(as.vector(density$z), decreasing = TRUE)
  cumulative <- cumsum(z) / sum(z)
  unique(sort(vapply(probabilities, function(probability) {
    z[which(cumulative >= probability)[1]]
  }, numeric(1))))
}

col_wave <- "#0072B2"
col_surge <- "#D55E00"
col_green <- "#009E73"
col_purple <- "#CC79A7"

pdf_device <- if (capabilities("cairo")) {
  grDevices::cairo_pdf
} else {
  grDevices::pdf
}
pdf_device(figure_file, width = 10, height = 8)

##
# Page 1: recursive evolution, proper scores, and optimization

par(mfrow = c(2, 2), mar = c(4, 4, 2.5, 1))
fan_plot(
  dates, observed[, 1], samples[, , 1],
  "Significant wave height (m)",
  "Recursive test-winter ensemble", col_wave
)
fan_plot(
  dates, observed[, 2], samples[, , 2],
  "Daily maximum surge (m)",
  "Recursive test-winter ensemble", col_surge
)

score_relative <- cbind(
  energy_score_standardized / stats::median(energy_score_standardized),
  variogram_score / stats::median(variogram_score)
)
score_range <- range(score_relative, finite = TRUE)
plot(
  dates, score_relative[, 1], type = "l", col = col_green, lwd = 1.2,
  xlab = "Date", ylab = "Score / median score",
  main = "Standardized multivariate scores",
  ylim = score_range
)
lines(dates, score_relative[, 2], col = col_purple, lwd = 1.2)
grid()
legend(
  "topleft",
  legend = c("Energy score", "Variogram score (p = 0.5)"),
  col = c(col_green, col_purple), lty = 1, lwd = 1.2,
  bty = "n", cex = 0.8
)

train_loss <- as.numeric(results$train_loss_history)
validation_loss <- as.numeric(results$val_loss_history)
n_epoch <- max(length(train_loss), length(validation_loss))
loss_matrix <- matrix(NA_real_, nrow = n_epoch, ncol = 2L)
loss_matrix[seq_along(train_loss), 1] <- train_loss
loss_matrix[seq_along(validation_loss), 2] <- validation_loss
matplot(
  seq_len(n_epoch), loss_matrix,
  type = "l", lty = 1, lwd = 1.3,
  col = c("black", col_surge),
  xlab = "Epoch", ylab = "Negative log likelihood",
  main = "Optimization history"
)
grid()
legend(
  "topright", legend = c("Training", "Validation"),
  col = c("black", col_surge), lty = 1, lwd = 1.3,
  bty = "n", cex = 0.8
)

##
# Page 2: marginal calibration

par(mfrow = c(2, 2), mar = c(4, 4, 2.5, 1))
pooled_rank_plot(
  marginal_ranks[, 1], "Wave-height rank histogram", col_wave
)
pooled_rank_plot(
  marginal_ranks[, 2], "Surge rank histogram", col_surge
)

matplot(
  coverage_levels, coverage,
  type = "b", pch = c(16, 17), lty = 1, lwd = 1.2,
  col = c(col_wave, col_surge),
  xlim = c(0.45, 0.95), ylim = c(0.35, 1),
  xlab = "Nominal central coverage", ylab = "Empirical coverage",
  main = "Central prediction intervals"
)
abline(0, 1, col = "grey40", lty = 2)
grid()
legend(
  "topleft", legend = c("Wave height", "Surge"),
  col = c(col_wave, col_surge), pch = c(16, 17),
  lty = 1, bty = "n", cex = 0.8
)

matplot(
  probability_levels, quantile_calibration,
  type = "l", lty = 1, lwd = 1.4,
  col = c(col_wave, col_surge),
  xlab = "Forecast quantile probability",
  ylab = "Fraction of observations below forecast quantile",
  main = "Marginal quantile calibration",
  xlim = c(0, 1), ylim = c(0, 1)
)
abline(0, 1, col = "grey40", lty = 2)
grid()
legend(
  "topleft", legend = c("Wave height", "Surge"),
  col = c(col_wave, col_surge), lty = 1, lwd = 1.4,
  bty = "n", cex = 0.8
)

##
# Page 3: multivariate calibration

par(
  mfrow = c(2, 2), mar = c(4, 4, 2.5, 1),
  oma = c(0, 0, 2, 0)
)
rank_colours <- c(col_green, col_wave, col_surge, col_purple)
for (j in seq_len(ncol(multivariate_ranks))) {
  pooled_rank_plot(
    multivariate_ranks[, j],
    colnames(multivariate_ranks)[j],
    rank_colours[j]
  )
}
mtext(
  "Multivariate rank histograms on the standardized model scale",
  outer = TRUE, cex = 1.1
)

##
# Page 4: dependence and seasonal extremes

par(mfrow = c(2, 2), mar = c(4, 4, 2.5, 1))
plot(
  observed[, 1], observed[, 2],
  pch = 16, cex = 0.55,
  col = grDevices::adjustcolor("black", alpha.f = 0.45),
  xlab = "Significant wave height (m)",
  ylab = "Daily maximum surge (m)",
  main = "Pooled joint distribution",
  xlim = joint_limits[1:2], ylim = joint_limits[3:4]
)
contour(
  forecast_density$x, forecast_density$y, forecast_density$z,
  levels = hdr_levels(forecast_density),
  add = TRUE, drawlabels = FALSE, col = col_wave, lwd = 1.5
)
contour(
  observed_density$x, observed_density$y, observed_density$z,
  levels = hdr_levels(observed_density),
  add = TRUE, drawlabels = FALSE, col = "black", lty = 2, lwd = 1.2
)
grid()
legend(
  "topleft",
  legend = c("Observed HDR contours", "Ensemble HDR contours"),
  col = c("black", col_wave), lty = c(2, 1), lwd = c(1.2, 1.5),
  bty = "n", cex = 0.8
)

hist(
  ensemble_correlation,
  breaks = "FD", col = col_wave, border = "white",
  xlab = "Within-member wave-surge correlation",
  main = "Whole-winter dependence",
  xlim = range(c(ensemble_correlation, observed_correlation))
)
abline(v = observed_correlation, col = "black", lwd = 2)
legend(
  "topleft",
  legend = sprintf("Observed: %.2f", observed_correlation),
  col = "black", lty = 1, lwd = 2, bty = "n", cex = 0.8
)

rolling_dates <- dates[rolling_index]
plot(
  rolling_dates, rolling_observed_correlation,
  type = "n", ylim = c(-1, 1),
  xlab = "Window end date", ylab = "Correlation",
  main = paste0(rolling_window, "-day rolling dependence")
)
polygon(
  c(rolling_dates, rev(rolling_dates)),
  c(
    rolling_correlation_quantiles[, 1],
    rev(rolling_correlation_quantiles[, 3])
  ),
  border = NA, col = grDevices::adjustcolor(col_wave, alpha.f = 0.18)
)
lines(
  rolling_dates, rolling_correlation_quantiles[, 2],
  col = col_wave, lwd = 1.4
)
lines(
  rolling_dates, rolling_observed_correlation,
  col = "black", lwd = 1.2
)
abline(h = 0, col = "grey50", lty = 3)
grid()
legend(
  "bottomleft",
  legend = c("Observed", "Ensemble median", "Ensemble 5%-95%"),
  col = c("black", col_wave, grDevices::adjustcolor(col_wave, alpha.f = 0.3)),
  lty = c(1, 1, NA), pch = c(NA, NA, 15),
  bty = "n", cex = 0.8
)

seasonal_maximum <- t(vapply(seq_len(n_ens), function(member) {
  apply(samples[, member, ], 2, max)
}, numeric(n_response)))
observed_maximum <- apply(observed, 2, max)
plot(
  seasonal_maximum[, 1], seasonal_maximum[, 2],
  pch = 16, col = grDevices::adjustcolor(col_wave, alpha.f = 0.55),
  xlab = "Winter maximum wave height (m)",
  ylab = "Winter maximum surge (m)",
  main = "Recursive seasonal maxima",
  xlim = range(c(seasonal_maximum[, 1], observed_maximum[1])),
  ylim = range(c(seasonal_maximum[, 2], observed_maximum[2]))
)
points(
  observed_maximum[1], observed_maximum[2],
  pch = 8, cex = 1.6, lwd = 2, col = "black"
)
grid()
legend(
  "topleft", legend = c("Ensemble members", "Observed winter"),
  col = c(col_wave, "black"), pch = c(16, 8),
  pt.cex = c(1, 1.4), bty = "n", cex = 0.8
)

##
# Page 5: temporal structure and spread-error relation

par(mfrow = c(2, 2), mar = c(4, 4, 2.5, 1))
for (j in seq_len(n_response)) {
  acf_quantiles <- t(apply(
    acf_ensemble[, , j], 1, quantile,
    probs = c(0.05, 0.5, 0.95), type = 8, names = FALSE
  ))
  plot(
    seq_len(lag_max), acf_observed[, j],
    type = "n", ylim = range(c(acf_observed[, j], acf_quantiles)),
    xlab = "Lag (days)", ylab = "Autocorrelation",
    main = paste(response_names[j], "persistence")
  )
  polygon(
    c(seq_len(lag_max), rev(seq_len(lag_max))),
    c(acf_quantiles[, 1], rev(acf_quantiles[, 3])),
    border = NA,
    col = grDevices::adjustcolor(
      c(col_wave, col_surge)[j], alpha.f = 0.18
    )
  )
  lines(
    seq_len(lag_max), acf_quantiles[, 2],
    col = c(col_wave, col_surge)[j], lwd = 1.4
  )
  lines(seq_len(lag_max), acf_observed[, j], col = "black", lwd = 1.2)
  abline(h = 0, col = "grey50", lty = 3)
  grid()
}

cross_quantiles <- t(apply(
  cross_ensemble, 1, quantile,
  probs = c(0.05, 0.5, 0.95), type = 8, names = FALSE
))
plot(
  cross_lags, cross_observed,
  type = "n", ylim = range(c(cross_observed, cross_quantiles)),
  xlab = "Lag k (days)",
  ylab = "cor[wave(t), surge(t + k)]",
  main = "Lagged wave-surge dependence"
)
polygon(
  c(cross_lags, rev(cross_lags)),
  c(cross_quantiles[, 1], rev(cross_quantiles[, 3])),
  border = NA, col = grDevices::adjustcolor(col_green, alpha.f = 0.18)
)
lines(cross_lags, cross_quantiles[, 2], col = col_green, lwd = 1.4)
lines(cross_lags, cross_observed, col = "black", lwd = 1.2)
abline(h = 0, col = "grey50", lty = 3)
grid()

spread_range <- range(unlist(lapply(spread_skill, function(x) x[, 1])))
rmse_range <- range(unlist(lapply(spread_skill, function(x) x[, 2])))
plot(
  spread_range, rmse_range, type = "n",
  xlab = "Mean ensemble SD (standardized scale)",
  ylab = "RMSE of ensemble mean (standardized scale)",
  main = "Binned spread-error relation"
)
for (j in seq_len(n_response)) {
  lines(
    spread_skill[[j]][, 1], spread_skill[[j]][, 2],
    type = "b", pch = c(16, 17)[j],
    col = c(col_wave, col_surge)[j], lwd = 1.2
  )
}
abline(0, 1, col = "grey40", lty = 2)
grid()
legend(
  "topleft", legend = c("Wave height", "Surge"),
  col = c(col_wave, col_surge), pch = c(16, 17),
  lty = 1, bty = "n", cex = 0.8
)

dev.off()

cat("Diagnostics written to:", figure_file, "\n")
cat("Verification dates:", format(min(dates)), "to", format(max(dates)), "\n")
cat("Cases:", n_test, "; ensemble members:", n_ens, "\n")
cat(
  "Mean energy score (physical response scale):",
  mean(energy_score_physical), "\n"
)
cat(
  "Mean energy score (standardized response scale):",
  mean(energy_score_standardized), "\n"
)
cat(
  "Mean CRPS: wave =", mean(crps[, 1]),
  "; surge =", mean(crps[, 2]), "\n"
)
cat(
  "Observed wave-surge correlation:", observed_correlation,
  "; mean ensemble-member correlation:", mean(ensemble_correlation), "\n"
)
cat("Central-interval coverage:\n")
print(round(coverage, 3))

################################################################################
