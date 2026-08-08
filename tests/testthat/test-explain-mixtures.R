test_that("mixture exceedance accounting separates weight and severity", {
  pred <- make_mdn_output(
    pi = matrix(c(0.8, 0.2), 1, 2),
    mu = array(c(-5, 5), c(1, 2, 1)),
    scale_chol = array(0.05, c(1, 2, 1, 1)),
    nu = matrix(Inf, 1, 2),
    alpha = array(0, c(1, 2, 1)),
    skew_none = TRUE
  )
  result <- suppressWarnings(tail_components_mst_pmdn(
    pred,
    response = 1L,
    threshold = 0,
    num_samples = 2048L,
    seed = 41
  ))
  expect_equal(result$data$component_probability, c(0, 1), tolerance = 0)
  expect_equal(result$data$contribution, c(0, 0.2), tolerance = 1e-6)
  expect_equal(unique(result$data$total_probability), 0.2, tolerance = 1e-6)
  expect_equal(result$data$tail_share, c(0, 1), tolerance = 1e-6)
  expect_equal(result$data$rank, c(2L, 1L))
  expect_false(".cache" %in% names(result$latent_draws))
})

test_that("tail-component rows retain one-based component identities", {
  pred <- make_mdn_output(
    pi = matrix(c(0.2, 0.3, 0.5), 1, 3),
    mu = array(c(-1, 0, 1), c(1, 3, 1)),
    scale_chol = array(1, c(1, 3, 1, 1)),
    nu = matrix(Inf, 1, 3),
    alpha = array(0, c(1, 3, 1)),
    skew_none = TRUE
  )
  result <- suppressWarnings(tail_components_mst_pmdn(
    pred, 1L, threshold = 0, num_samples = 256L, seed = 42
  ))
  expect_identical(result$data$component, 1:3)
  expect_equal(
    unique(result$data$total_probability),
    sum(result$data$contribution),
    tolerance = 1e-8
  )
})
