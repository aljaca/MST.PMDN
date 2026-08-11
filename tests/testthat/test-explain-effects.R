test_that("linear-location ALE has the known slope and empirical centring", {
  x <- cbind(feature = seq(-2, 2, length.out = 101), other = 0)
  model <- explanation_test_model(slope = 2.5)
  result <- ale_mst_pmdn(
    model,
    inputs = x,
    feature = "feature",
    functional = mst_functional("mean", 1L),
    n_bins = 10L,
    chunk_size = 7L
  )
  fitted_slope <- unname(stats::coef(stats::lm(
    ale ~ feature_value, data = result$data
  ))[2L])
  expect_equal(fitted_slope, 2.5, tolerance = 1e-5)
  expect_equal(
    stats::weighted.mean(result$data$ale, result$data$n),
    0,
    tolerance = 1e-7
  )
  expect_equal(sum(result$data$n), nrow(x))
})

test_that("ALE chunking preserves results", {
  x <- cbind(feature = seq(-1, 1, length.out = 37), other = 1)
  model <- explanation_test_model(slope = -1.5)
  functional <- mst_functional("quantile", 1L, prob = 0.9)
  bank <- latent_draws_mst_pmdn(512L, output_dim = 1L, seed = 21)
  whole <- suppressWarnings(ale_mst_pmdn(
    model, x, 1L, functional,
    n_bins = 6L,
    latent_draws = bank,
    chunk_size = nrow(x)
  ))
  chunked <- suppressWarnings(ale_mst_pmdn(
    model, x, 1L, functional,
    n_bins = 6L,
    latent_draws = bank,
    chunk_size = 3L
  ))
  expect_equal(whole$data$ale, chunked$data$ale, tolerance = 0)
  expect_equal(whole$data$n, chunked$data$n)
})

test_that("ALE tail diagnostics use evaluated event probabilities", {
  x <- cbind(feature = seq(-1, 1, length.out = 21), other = 0)
  model <- explanation_test_model(slope = 1)
  functional <- mst_functional(
    "exceedance", 1L, threshold = 10, direction = "upper"
  )
  bank <- latent_draws_mst_pmdn(128L, output_dim = 1L, seed = 22)
  result <- ale_mst_pmdn(
    model,
    x,
    1L,
    functional,
    n_bins = 5L,
    latent_draws = bank
  )
  expect_equal(result$diagnostics$min_expected_tail_draws, 0)
  expect_equal(
    result$diagnostics$min_expected_tail_draws_by_bin,
    rep(0, nrow(result$data))
  )
})

test_that("tabular perturbations retain case-matched image rows", {
  x <- cbind(feature = c(0, 0.2, 0.4, 0.6, 0.8, 1), other = 0)
  image <- array(0, c(nrow(x), 1, 1, 1))
  image[, 1, 1, 1] <- c(1, 3, 5, 2, 4, 8)
  model <- explanation_test_model(slope = 1, image_interaction = TRUE)
  result <- ale_mst_pmdn(
    model,
    x,
    feature = 1L,
    image_inputs = image,
    functional = mst_functional("mean", 1L),
    n_bins = 3L,
    chunk_size = 2L
  )

  breaks <- result$breaks
  bins <- pmin(
    findInterval(x[, 1], breaks, all.inside = TRUE),
    length(breaks) - 1L
  )
  local <- vapply(seq_len(length(breaks) - 1L), function(k) {
    rows <- which(bins == k)
    (breaks[k + 1L] - breaks[k]) * mean(image[rows, 1, 1, 1])
  }, numeric(1))
  midpoint <- cumsum(local) - 0.5 * local
  expected <- midpoint - stats::weighted.mean(midpoint, result$data$n)
  expect_equal(result$data$ale, expected, tolerance = 1e-6)
})

test_that("centred ICE includes the known effect and an ALE overlay", {
  x <- cbind(feature = seq(-1, 1, length.out = 21), other = 0)
  model <- explanation_test_model(slope = 3)
  bank <- latent_draws_mst_pmdn(128L, output_dim = 1L, seed = 23)
  result <- ice_mst_pmdn(
    model,
    x,
    feature = "feature",
    functional = mst_functional("quantile", 1L, prob = 0.8),
    grid = seq(-1, 1, length.out = 7),
    reference = 0,
    n_curves = 5L,
    n_bins = 5L,
    latent_draws = bank
  )
  expect_equal(
    result$curves$centred,
    3 * result$curves$feature_value,
    tolerance = 1e-6
  )
  expect_identical(
    names(result$curves),
    c("case", "feature_value", "value", "centred")
  )
  expect_s3_class(result$ale, "mst_pmdn_ale")
  expect_false(".cache" %in% names(result$latent_draws))
  expect_false(".cache" %in% names(result$ale$latent_draws))
})

test_that("ALE merges empirically empty bins rather than accumulating NaN", {
  # Type-7 quartiles are 0, 0.75, 1, 1.25, 2. Because values equal to an
  # interior boundary enter the upper interval, the (0.75, 1) bin is empty.
  x <- cbind(feature = c(0, 1, 1, 2), other = 0)
  result <- ale_mst_pmdn(
    explanation_test_model(slope = 2),
    x,
    feature = 1L,
    functional = mst_functional("mean", 1L),
    n_bins = 4L
  )
  expect_equal(result$settings$actual_bins, 3L)
  expect_true(all(result$data$n > 0L))
  expect_true(all(is.finite(result$data$ale)))
  expect_equal(sum(result$data$n), nrow(x))
})

test_that("finite skewed mixture ALE reaches the complete sampling path", {
  x <- cbind(feature = seq(-1, 1, length.out = 8), other = 0)
  image <- array(seq(0.1, 0.8, length.out = 8), c(8, 1, 1, 1))
  model <- distribution_explanation_test_model(n_mixtures = 2L)
  functional <- mst_functional(
    "joint_exceedance", c(1L, 2L), threshold = c(0, 0)
  )
  bank <- latent_draws_mst_pmdn(128L, output_dim = 2L, seed = 61)
  result <- suppressWarnings(ale_mst_pmdn(
    model,
    x,
    feature = 1L,
    image_inputs = image,
    functional = functional,
    n_bins = 3L,
    latent_draws = bank,
    chunk_size = 2L
  ))
  expect_true(all(is.finite(result$data$ale)))
  expect_equal(sum(result$data$n), nrow(x))
  expect_false(".cache" %in% names(result$latent_draws))
})

test_that("multi-channel ALE decomposition integrates Shapley bin effects", {
  x <- cbind(feature = seq(-1, 1, length.out = 6), other = 0)
  model <- distribution_explanation_test_model(n_mixtures = 1L)
  bank <- latent_draws_mst_pmdn(128L, output_dim = 2L, seed = 62)
  result <- suppressWarnings(ale_mst_pmdn(
    model,
    x,
    feature = 1L,
    functional = mst_functional("quantile", 1L, prob = 0.8),
    n_bins = 2L,
    decompose = TRUE,
    latent_draws = bank,
    chunk_size = 2L
  ))
  expect_setequal(
    result$active_channels,
    c("location", "scale", "skewness", "df")
  )
  expect_equal(result$data$sum_to_total_residual, rep(0, nrow(result$data)),
               tolerance = 1e-9)
  expect_true(any(abs(result$data$ale_scale) > 0))
  expect_true(any(abs(result$data$ale_skewness) > 0))
  expect_true(any(abs(result$data$ale_df) > 0))
  expect_false(".cache" %in% names(result$latent_draws))
})

test_that("ALE decomposition rejects mixtures before bin evaluation", {
  x <- cbind(feature = seq(-1, 1, length.out = 6), other = 0)
  expect_error(
    ale_mst_pmdn(
      distribution_explanation_test_model(n_mixtures = 2L),
      x,
      feature = 1L,
      functional = mst_functional("mean", 1L),
      n_bins = 2L,
      decompose = TRUE
    ),
    "only available for M = 1"
  )
})

test_that("ICE can omit or reuse its population ALE overlay", {
  x <- cbind(feature = seq(-1, 1, length.out = 11), other = 0)
  model <- explanation_test_model(slope = 2)
  functional <- mst_functional("mean", 1L)
  precomputed <- ale_mst_pmdn(
    model, x, 1L, functional, n_bins = 3L
  )
  reused <- ice_mst_pmdn(
    model,
    x,
    1L,
    functional,
    grid = c(-1, 0, 1),
    n_curves = 3L,
    ale = precomputed
  )
  omitted <- ice_mst_pmdn(
    model,
    x,
    1L,
    functional,
    grid = c(-1, 0, 1),
    n_curves = 3L,
    ale = FALSE
  )
  expect_identical(reused$ale, precomputed)
  expect_identical(reused$settings$ale, "supplied")
  expect_null(omitted$ale)
  expect_identical(omitted$settings$ale, "none")
})

test_that("channel-specific ALE closes to the total effect", {
  x <- cbind(feature = seq(-1, 1, length.out = 31), other = 0)
  model <- explanation_test_model(slope = 2)
  result <- ale_mst_pmdn(
    model,
    x,
    feature = 1L,
    functional = mst_functional("mean", 1L),
    n_bins = 6L,
    decompose = TRUE,
    chunk_size = 4L
  )
  expect_identical(result$active_channels, "location")
  expect_equal(result$data$ale_location, result$data$ale, tolerance = 1e-7)
  expect_equal(
    result$data$sum_to_total_residual,
    rep(0, nrow(result$data)),
    tolerance = 1e-7
  )
})

test_that("ICE accepts explicit one-based case indices", {
  x <- cbind(feature = seq(-1, 1, length.out = 9), other = 0)
  result <- ice_mst_pmdn(
    explanation_test_model(slope = 2),
    x,
    feature = 1L,
    functional = mst_functional("mean", 1L),
    grid = c(-1, 0, 1),
    cases = c(9L, 2L, 5L),
    ale = FALSE
  )
  expect_identical(result$cases, c(9L, 2L, 5L))
  expect_identical(unique(result$curves$case), c(9L, 2L, 5L))
  expect_identical(result$settings$case_selection, "explicit")
})

test_that("ALE and nested ICE each aggregate tail warnings once", {
  x <- cbind(feature = seq(-1, 1, length.out = 6), other = 0)
  model <- distribution_explanation_test_model(n_mixtures = 1L)
  functional <- mst_functional(
    "joint_exceedance",
    c(1L, 2L),
    threshold = c(1e30, 1e30)
  )
  bank <- latent_draws_mst_pmdn(64L, output_dim = 2L, seed = 63)
  calls <- list(
    ALE = function() ale_mst_pmdn(
      model,
      x,
      feature = 1L,
      functional = functional,
      n_bins = 2L,
      latent_draws = bank,
      min_tail_draws = 20L
    ),
    ICE = function() ice_mst_pmdn(
      model,
      x,
      feature = 1L,
      functional = functional,
      grid = c(-1, 1),
      n_curves = 2L,
      ale = TRUE,
      n_bins = 2L,
      latent_draws = bank,
      min_tail_draws = 20L
    )
  )
  for (name in names(calls)) {
    warning_count <- 0L
    result <- withCallingHandlers(
      calls[[name]](),
      mst_pmdn_tail_resolution_warning = function(condition) {
        warning_count <<- warning_count + 1L
        invokeRestart("muffleWarning")
      }
    )
    expect_equal(warning_count, 1L, info = name)
    expect_equal(result$diagnostics$min_expected_tail_draws, 0)
    expect_gt(result$diagnostics$low_tail_resolution_evaluations, 0)
  }
})
