test_that("all explanation plot methods smoke-test with graphical overrides", {
  grDevices::pdf(NULL)
  on.exit(grDevices::dev.off(), add = TRUE)

  pred <- make_mdn_output(
    pi = matrix(1, 1, 1),
    mu = array(0, c(1, 1, 1)),
    scale_chol = array(1, c(1, 1, 1, 1)),
    nu = matrix(Inf, 1, 1),
    alpha = array(0, c(1, 1, 1)),
    skew_none = TRUE
  )
  functional_value <- functional_mst_pmdn(
    pred, mst_functional("mean", 1L)
  )
  expect_invisible(plot(
    functional_value,
    main = "Functional",
    xlab = "Row",
    ylab = "Value"
  ))

  x <- cbind(feature = seq(-1, 1, length.out = 9), other = 0)
  model <- explanation_test_model(slope = 2)
  ale <- ale_mst_pmdn(
    model,
    x,
    feature = 1L,
    functional = mst_functional("mean", 1L),
    n_bins = 3L,
    decompose = TRUE
  )
  expect_invisible(plot(
    ale,
    type = "total",
    main = "ALE",
    xlab = "Predictor",
    ylab = "Effect"
  ))
  expect_invisible(plot(ale, type = "channels", main = "ALE channels"))

  ice <- ice_mst_pmdn(
    model,
    x,
    feature = 1L,
    functional = mst_functional("mean", 1L),
    grid = c(-1, 0, 1),
    n_curves = 3L,
    derivative = TRUE,
    ale = FALSE
  )
  expect_invisible(plot(
    ice,
    type = "ice",
    main = "ICE",
    xlab = "Predictor",
    ylab = "Centred effect"
  ))
  expect_invisible(plot(
    ice,
    type = "plate",
    main = "Plate",
    xlab = "Baseline",
    ylab = "Slope",
    pch = 1
  ))

  pred_to <- pred
  pred_to$mu <- torch::torch_tensor(array(1, c(1, 1, 1)))
  decomposition <- decompose_mst_pmdn(
    pred, pred_to, mst_functional("mean", 1L)
  )
  expect_invisible(plot(
    decomposition,
    main = "Decomposition",
    xlab = "Channel",
    ylab = "Contribution"
  ))

  image <- array(2, c(1, 1, 2, 2))
  reference <- array(0, c(1, 1, 2, 2))
  image_contrast <- image_contrast_mst_pmdn(
    explanation_test_model(slope = 0),
    matrix(0, 1, 1),
    image,
    reference,
    mst_functional("mean", 1L),
    decompose = TRUE
  )
  expect_invisible(plot(
    image_contrast,
    type = "distribution",
    main = "Contrast distribution",
    xlab = "Contrast",
    ylab = "Count"
  ))
  expect_invisible(plot(
    image_contrast,
    type = "cases",
    main = "Case contrasts",
    xlab = "Case",
    ylab = "Contrast"
  ))
  expect_invisible(plot(
    image_contrast,
    type = "channels",
    main = "Image channels",
    xlab = "Channel",
    ylab = "Contribution"
  ))

  irregular_image <- array(1, c(1, 1, 8, 8))
  irregular_occlusion <- image_occlusion_mst_pmdn(
    explanation_test_model(slope = 0),
    matrix(0, 1, 1),
    irregular_image,
    array(0, c(1, 1, 8, 8)),
    mst_functional("mean", 1L),
    patch_size = c(3L, 3L),
    stride = c(2L, 2L),
    taper = "none"
  )
  row_centres <- sort(unique(irregular_occlusion$patches$row_center))
  expect_gt(length(unique(diff(row_centres))), 1L)
  expect_invisible(plot(
    irregular_occlusion,
    main = "Irregular occlusion grid",
    xlab = "Column",
    ylab = "Row",
    col = c("navy", "white", "firebrick"),
    zlim = c(-2, 2)
  ))

  mixture <- make_mdn_output(
    pi = matrix(c(0.7, 0.3), 1, 2),
    mu = array(c(-1, 1), c(1, 2, 1)),
    scale_chol = array(1, c(1, 2, 1, 1)),
    nu = matrix(Inf, 1, 2),
    alpha = array(0, c(1, 2, 1)),
    skew_none = TRUE
  )
  tail_sources <- suppressWarnings(tail_components_mst_pmdn(
    mixture, 1L, threshold = 0, num_samples = 64L, seed = 81
  ))
  expect_invisible(plot(
    tail_sources,
    row = 1L,
    main = "Roberts Bank",
    ylab = "Probability",
    xlab = "Component"
  ))
})
