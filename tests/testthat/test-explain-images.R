test_that("whole-image contrasts use case-specific references", {
  x <- matrix(0, nrow = 2, ncol = 1)
  image <- array(0, c(2, 1, 2, 2))
  image[1, 1, , ] <- 2
  image[2, 1, , ] <- 5
  reference <- array(1, c(1, 1, 2, 2))
  model <- explanation_test_model(slope = 0, image_channel = 1L)
  bank <- latent_draws_mst_pmdn(128L, output_dim = 1L, seed = 73)
  result <- image_contrast_mst_pmdn(
    model,
    x,
    image,
    reference,
    mst_functional("quantile", 1L, prob = 0.8),
    chunk_size = 1L,
    latent_draws = bank
  )
  expect_equal(result$data$contrast, c(1, 4), tolerance = 1e-6)
  expect_false(".cache" %in% names(result$latent_draws))
})

test_that("reference images are coerced to the input tensor representation", {
  x <- matrix(0, nrow = 1, ncol = 1)
  image <- torch::torch_tensor(array(3, c(1, 1, 2, 2)))
  reference <- array(1, c(1, 1, 2, 2))
  model <- explanation_test_model(slope = 0, image_channel = 1L)
  result <- image_contrast_mst_pmdn(
    model,
    x,
    image,
    reference,
    mst_functional("mean", 1L)
  )
  expect_equal(result$data$contrast, 2, tolerance = 1e-6)
})

test_that("rebuild_channels restores deterministic derived image channels", {
  x <- matrix(0, nrow = 1, ncol = 1)
  pressure <- array(4, c(1, 1, 2, 2))
  reference <- array(1, c(1, 1, 2, 2))
  rebuild <- function(base_images, reference_images, masks,
                      case_index = NULL) {
    pressure_masked <- (1 - masks) * base_images + masks * reference_images
    torch::torch_cat(
      list(pressure_masked, 2 * pressure_masked), dim = 2L
    )
  }
  pressure_tensor <- torch::torch_tensor(pressure)
  reference_tensor <- torch::torch_tensor(reference)
  model <- explanation_test_model(slope = 0, image_channel = 2L)
  result <- image_contrast_mst_pmdn(
    model,
    x,
    pressure_tensor,
    reference_tensor,
    mst_functional("mean", 1L),
    rebuild_channels = rebuild
  )
  expect_equal(result$data$contrast, 6, tolerance = 1e-6)
})

test_that("rebuild_channels must return compatible image states", {
  x <- matrix(0, nrow = 1, ncol = 1)
  image <- array(2, c(1, 1, 2, 2))
  reference <- array(0, c(1, 1, 2, 2))
  rebuild <- function(base_images, reference_images, masks,
                      case_index = NULL) {
    blended <- (1 - masks) * base_images + masks * reference_images
    if (sum(masks) > 0) {
      blended[, , 1L, , drop = FALSE]
    } else {
      blended
    }
  }
  model <- explanation_test_model(slope = 0, image_channel = 1L)
  expect_error(
    image_contrast_mst_pmdn(
      model,
      x,
      image,
      reference,
      mst_functional("mean", 1L),
      rebuild_channels = rebuild
    ),
    "must match the rebuilt original image shape"
  )
})

test_that("full-patch occlusion matches the whole-image contrast", {
  x <- matrix(0, nrow = 2, ncol = 1)
  image <- array(0, c(2, 1, 2, 2))
  image[1, 1, , ] <- 3
  image[2, 1, , ] <- 6
  reference <- array(1, c(1, 1, 2, 2))
  model <- explanation_test_model(slope = 0, image_channel = 1L)
  result <- image_occlusion_mst_pmdn(
    model,
    x,
    image,
    reference,
    mst_functional("mean", 1L),
    patch_size = c(2L, 2L),
    stride = c(2L, 2L),
    taper = "none"
  )
  expect_equal(result$data$effect, c(2, 5), tolerance = 1e-6)
  expect_equal(result$population$mean_signed_effect, 3.5, tolerance = 1e-6)
  expect_true(all(result$coverage == 1))
})

test_that("image occlusion can return single-component channel maps", {
  x <- matrix(0, nrow = 1, ncol = 1)
  image <- array(2, c(1, 1, 2, 2))
  reference <- array(0, c(1, 1, 2, 2))
  model <- explanation_test_model(slope = 0, image_channel = 1L)
  result <- image_occlusion_mst_pmdn(
    model,
    x,
    image,
    reference,
    mst_functional("mean", 1L),
    patch_size = c(2L, 2L),
    stride = c(1L, 1L),
    taper = "none",
    decompose = TRUE
  )
  expect_identical(result$active_channels, "location")
  expect_equal(result$data$channel_location, result$data$effect,
               tolerance = 1e-6)
  expect_equal(result$data$sum_to_total_residual, 0, tolerance = 1e-10)
})

test_that("finite skewed mixture occlusion reaches complete functional sampling", {
  x <- cbind(feature = c(-0.25, 0.5), other = 0)
  image <- array(c(1, 2), c(2, 1, 1, 1))
  reference <- array(0, c(1, 1, 1, 1))
  model <- distribution_explanation_test_model(n_mixtures = 2L)
  bank <- latent_draws_mst_pmdn(128L, output_dim = 2L, seed = 71)
  result <- suppressWarnings(image_occlusion_mst_pmdn(
    model,
    x,
    image,
    reference,
    mst_functional(
      "joint_exceedance", c(1L, 2L), threshold = c(0, 0)
    ),
    patch_size = c(1L, 1L),
    stride = c(1L, 1L),
    taper = "none",
    latent_draws = bank,
    chunk_size = 1L
  ))
  expect_true(all(is.finite(result$data$effect)))
  expect_equal(nrow(result$data), 2L)
  expect_false(".cache" %in% names(result$latent_draws))
})

test_that("multi-channel image decomposition closes within each patch", {
  x <- cbind(feature = 0, other = 0)
  image <- array(1, c(1, 1, 2, 2))
  reference <- array(0, c(1, 1, 2, 2))
  model <- distribution_explanation_test_model(n_mixtures = 1L)
  bank <- latent_draws_mst_pmdn(128L, output_dim = 2L, seed = 72)
  result <- suppressWarnings(image_occlusion_mst_pmdn(
    model,
    x,
    image,
    reference,
    mst_functional("quantile", 1L, prob = 0.8),
    patch_size = c(2L, 2L),
    stride = c(1L, 1L),
    taper = "none",
    decompose = TRUE,
    latent_draws = bank
  ))
  expect_setequal(
    result$active_channels,
    c("location", "scale", "skewness", "df")
  )
  expect_equal(result$data$sum_to_total_residual, 0, tolerance = 1e-9)
  for (channel in result$active_channels) {
    expect_true(paste0("channel_", channel) %in% names(result$data))
  }
  expect_false(".cache" %in% names(result$latent_draws))
})

test_that("image decomposition rejects mixtures before patch evaluation", {
  x <- cbind(feature = 0, other = 0)
  image <- array(1, c(1, 1, 2, 2))
  reference <- array(0, c(1, 1, 2, 2))
  expect_error(
    image_occlusion_mst_pmdn(
      distribution_explanation_test_model(n_mixtures = 2L),
      x,
      image,
      reference,
      mst_functional("mean", 1L),
      patch_size = c(1L, 1L),
      stride = c(1L, 1L),
      decompose = TRUE
    ),
    "only available for M = 1"
  )
})
