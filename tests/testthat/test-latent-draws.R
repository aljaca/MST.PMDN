test_that("fixed latent banks are reproducible and preserve dtype", {
  first <- latent_draws_mst_pmdn(
    128L, output_dim = 2L, dtype = torch::torch_double(), seed = 17
  )
  second <- latent_draws_mst_pmdn(
    128L, output_dim = 2L, dtype = torch::torch_double(), seed = 17
  )
  expect_identical(as.character(first$component_u$dtype), "Double")
  expect_identical(as.character(first$skew_z$dtype), "Double")
  expect_equal(
    torch::as_array(first$component_u),
    torch::as_array(second$component_u),
    tolerance = 0
  )
  expect_equal(
    torch::as_array(first$gamma_u),
    torch::as_array(second$gamma_u),
    tolerance = 0
  )
})

test_that("latent seeds leave the R RNG stream unchanged", {
  set.seed(1234)
  before <- .Random.seed
  latent_draws_mst_pmdn(16L, output_dim = 1L, seed = 99)
  expect_identical(.Random.seed, before)
})

test_that("finite-df Gamma transforms reuse repeated selected df states", {
  pred <- make_mdn_output(
    pi = matrix(1, 1, 1),
    mu = array(0, c(1, 1, 1)),
    scale_chol = array(1, c(1, 1, 1, 1)),
    nu = matrix(7, 1, 1),
    alpha = array(0.3, c(1, 1, 1))
  )
  bank <- latent_draws_mst_pmdn(64L, output_dim = 1L, seed = 100)
  functional <- mst_functional("quantile", 1L, prob = 0.8)
  first <- suppressWarnings(functional_mst_pmdn(
    pred, functional, latent_draws = bank
  ))
  second <- suppressWarnings(functional_mst_pmdn(
    pred, functional, latent_draws = bank
  ))
  expect_equal(bank$.cache$gamma_scale_misses, 1L)
  expect_equal(bank$.cache$gamma_scale_hits, 1L)
  expect_false(".cache" %in% names(first$latent_draws))
  expect_false(".cache" %in% names(second$latent_draws))
})

test_that("finite-df Gamma caching follows mutations to gamma_u", {
  pred <- make_mdn_output(
    pi = matrix(1, 1, 1),
    mu = array(0, c(1, 1, 1)),
    scale_chol = array(1, c(1, 1, 1, 1)),
    nu = matrix(7, 1, 1),
    alpha = array(0.3, c(1, 1, 1))
  )
  bank <- latent_draws_mst_pmdn(64L, output_dim = 1L, seed = 101)
  functional <- mst_functional("quantile", 1L, prob = 0.8)
  suppressWarnings(functional_mst_pmdn(
    pred, functional, latent_draws = bank
  ))
  bank$gamma_u <- 1 - bank$gamma_u
  changed <- suppressWarnings(functional_mst_pmdn(
    pred, functional, latent_draws = bank
  ))
  expect_equal(bank$.cache$gamma_scale_misses, 2L)

  uncached <- bank
  uncached$.cache <- new.env(parent = emptyenv())
  expected <- suppressWarnings(functional_mst_pmdn(
    pred, functional, latent_draws = uncached
  ))
  expect_equal(changed$data$value, expected$data$value, tolerance = 0)
})

test_that("latent-bank sampling uses R torch one-based component indices", {
  pred <- make_mdn_output(
    pi = rbind(c(1, 0), c(0, 1)),
    mu = array(c(-2, 2, -2, 2), c(2, 2, 1)),
    scale_chol = array(1, c(2, 2, 1, 1)),
    nu = matrix(Inf, 2, 2),
    alpha = array(0, c(2, 2, 1)),
    skew_none = TRUE
  )
  bank <- latent_draws_mst_pmdn(50L, output_dim = 1L, seed = 2)
  sampled <- MST.PMDN:::.sample_with_latent_mst_pmdn(pred, bank)
  components <- torch::as_array(sampled$components)
  expect_true(all(components[, 1] == 1L))
  expect_true(all(components[, 2] == 2L))
})

test_that("chunked and unchunked Monte Carlo functionals are identical", {
  B <- 5L
  pred <- make_mdn_output(
    pi = matrix(1, B, 1),
    mu = array(seq(-1, 1, length.out = B), c(B, 1, 1)),
    scale_chol = array(1, c(B, 1, 1, 1)),
    nu = matrix(7, B, 1),
    alpha = array(0.4, c(B, 1, 1))
  )
  bank <- latent_draws_mst_pmdn(2048L, output_dim = 1L, seed = 3)
  functional <- mst_functional("quantile", 1L, prob = 0.9)
  whole <- suppressWarnings(functional_mst_pmdn(
    pred, functional, latent_draws = bank, chunk_size = B
  ))
  chunked <- suppressWarnings(functional_mst_pmdn(
    pred, functional, latent_draws = bank, chunk_size = 2L
  ))
  expect_equal(whole$data$value, chunked$data$value, tolerance = 0)
})

test_that("latent evaluation supports float32, float64, and conditional CUDA", {
  for (dtype in list(torch::torch_float(), torch::torch_double())) {
    pred <- make_mdn_output(
      pi = matrix(1, 1, 1),
      mu = array(0, c(1, 1, 1)),
      scale_chol = array(1, c(1, 1, 1, 1)),
      nu = matrix(Inf, 1, 1),
      alpha = array(0, c(1, 1, 1)),
      skew_none = TRUE,
      dtype = dtype
    )
    bank <- latent_draws_mst_pmdn(
      32L, output_dim = 1L, dtype = dtype, seed = 4
    )
    sampled <- MST.PMDN:::.sample_with_latent_mst_pmdn(pred, bank)
    expect_identical(
      as.character(sampled$samples$dtype),
      as.character(dtype)
    )
  }

  skip_if_not(torch::cuda_is_available(), "CUDA is not available")
  pred <- make_mdn_output(
    pi = matrix(1, 1, 1),
    mu = array(0, c(1, 1, 1)),
    scale_chol = array(1, c(1, 1, 1, 1)),
    nu = matrix(Inf, 1, 1),
    alpha = array(0, c(1, 1, 1)),
    skew_none = TRUE,
    device = "cuda"
  )
  bank <- latent_draws_mst_pmdn(
    32L, output_dim = 1L, device = "cuda", seed = 5
  )
  sampled <- MST.PMDN:::.sample_with_latent_mst_pmdn(
    pred, bank, device = "cuda"
  )
  expect_true(sampled$samples$device$type == "cuda")
})
