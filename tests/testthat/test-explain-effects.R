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

test_that("centred ICE includes ALE, derivatives, and Plate data", {
  x <- cbind(feature = seq(-1, 1, length.out = 21), other = 0)
  model <- explanation_test_model(slope = 3)
  result <- ice_mst_pmdn(
    model,
    x,
    feature = "feature",
    functional = mst_functional("mean", 1L),
    grid = seq(-1, 1, length.out = 7),
    reference = 0,
    n_curves = 5L,
    derivative = TRUE,
    n_bins = 5L
  )
  expect_equal(
    result$curves$slope,
    rep(3, nrow(result$curves)),
    tolerance = 1e-6
  )
  expect_equal(result$plate$local_slope, rep(3, nrow(result$plate)),
               tolerance = 1e-6)
  expect_s3_class(result$ale, "mst_pmdn_ale")
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
