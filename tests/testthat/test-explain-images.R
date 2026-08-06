test_that("whole-image contrasts use case-specific references", {
  x <- matrix(0, nrow = 2, ncol = 1)
  image <- array(0, c(2, 1, 2, 2))
  image[1, 1, , ] <- 2
  image[2, 1, , ] <- 5
  reference <- array(1, c(1, 1, 2, 2))
  model <- explanation_test_model(slope = 0, image_channel = 1L)
  result <- image_contrast_mst_pmdn(
    model,
    x,
    image,
    reference,
    mst_functional("mean", 1L),
    chunk_size = 1L
  )
  expect_equal(result$data$contrast, c(1, 4), tolerance = 1e-6)
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
