################################################################################
# Distribution-functional XAI for the fitted MST-PMDN wave-surge example
#
# This script is run after examples/wave-surge-dailymax.R. It reloads the
# fitted model and frozen preprocessing object; it does not refit the network.
# The explanations use one-step-ahead test predictors, for which the lagged
# responses are observed. They therefore explain the saved pred_test output,
# not the recursively generated ensemble trajectories.
#
# Usage:
#   Rscript wave-surge-xai.R
#   Rscript wave-surge-xai.R model.pt results.rds wave_surge.RData \
#     wave-surge-xai.pdf
################################################################################

library(torch)
library(MST.PMDN)

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) > 4L) {
    stop(
      "The XAI RDS output has been removed. Pass at most: model.pt, ",
      "results.rds, wave_surge.RData, and figures.pdf."
    )
  }
  argument_or <- function(index, default) {
    if (length(args) >= index && nzchar(args[index])) args[index] else default
  }

  model_file <- argument_or(
    1L, file.path("out", "wave-surge-dailymax.pt")
  )
  results_file <- argument_or(
    2L, file.path("out", "wave-surge-dailymax-results.rds")
  )
  data_file <- argument_or(3L, "wave_surge.RData")
  figure_file <- argument_or(
    4L, file.path("out", "wave-surge-dailymax-xai.pdf")
  )

  # Analysis settings. The two lag-1 predictors avoid incoherent independent
  # perturbations of the seasonal sine/cosine pairs. Higher response lags can
  # be added to interpreted_features without changing the workflow.
  threshold_probability <- 0.95
  interpreted_features <- c(
    "wave_transformed_lag1",
    "surge_lag1"
  )
  num_samples <- 8192L
  min_tail_draws <- 20L
  n_ale_bins <- 12L
  n_ice_grid_points <- 25L
  n_ice_display_curves <- 40L
  n_image_contrast_cases <- 20L
  n_occlusion_cases <- 5L
  n_occlusion_banks <- 3L
  patch_size <- c(8L, 8L)
  stride <- c(4L, 4L)
  chunk_size <- 16L
  prediction_tolerance <- 1e-5

  required_functions <- c(
    "latent_draws_mst_pmdn", "mst_functional", "functional_mst_pmdn",
    "ale_mst_pmdn", "ice_mst_pmdn", "image_contrast_mst_pmdn",
    "image_occlusion_mst_pmdn", "decompose_mst_pmdn",
    "tail_components_mst_pmdn"
  )
  missing_functions <- required_functions[!vapply(
    required_functions,
    exists,
    logical(1),
    envir = asNamespace("MST.PMDN"),
    inherits = FALSE
  )]
  if (length(missing_functions)) {
    stop(
      "The installed MST.PMDN does not contain the XAI branch functions: ",
      paste(missing_functions, collapse = ", "), "."
    )
  }

  for (path in c(model_file, results_file, data_file)) {
    if (!file.exists(path)) stop("Input file not found: ", path)
  }
  figure_parent <- dirname(figure_file)
  if (!dir.exists(figure_parent) &&
      !dir.create(figure_parent, recursive = TRUE, showWarnings = FALSE)) {
    stop("Could not create output directory: ", figure_parent)
  }

  device <- if (cuda_is_available()) "cuda" else "cpu"
  torch_set_num_threads(1L)
  torch_set_num_interop_threads(1L)

  results <- readRDS(results_file)
  required_results <- c(
    "seed", "n_mixtures", "train_seasons", "test_dates",
    "observed_test", "pred_test"
  )
  missing_items <- setdiff(required_results, names(results))
  if (length(missing_items)) {
    stop(
      "The fitting results are missing: ",
      paste(missing_items, collapse = ", "), "."
    )
  }
  if (!is.list(results$pred_test)) {
    stop("results$pred_test must be the saved one-step prediction list.")
  }

  xai_seed <- as.integer(results$seed) + 2L
  if (length(xai_seed) != 1L || is.na(xai_seed) || xai_seed < 0L) {
    stop("The fitting seed cannot be converted to a valid XAI seed.")
  }
  torch_manual_seed(xai_seed)

  data_environment <- new.env(parent = baseenv())
  loaded_objects <- load(data_file, envir = data_environment)
  if (!identical(loaded_objects, "wave_surge") ||
      !is.list(data_environment$wave_surge)) {
    stop(data_file, " must contain one list named 'wave_surge'.")
  }
  wave_surge <- data_environment$wave_surge
  rm(data_environment, loaded_objects)

  required_data <- c(
    "date", "y", "y_original", "x", "x_image", "y_mean", "y_sd",
    "x_image_mean", "x_image_sd", "wave_scale", "inv_softplus",
    "softplus"
  )
  missing_items <- setdiff(required_data, names(wave_surge))
  if (length(missing_items)) {
    stop(
      "The wave_surge object is missing: ",
      paste(missing_items, collapse = ", "), "."
    )
  }

  data_dates <- as.Date(wave_surge$date)
  data_y <- as.matrix(wave_surge$y)
  data_y_original <- as.matrix(wave_surge$y_original)
  data_x <- as.matrix(wave_surge$x)
  data_x_image <- wave_surge$x_image
  target_center <- as.numeric(wave_surge$y_mean)
  target_scale <- as.numeric(wave_surge$y_sd)
  image_center <- wave_surge$x_image_mean
  image_scale <- wave_surge$x_image_sd
  wave_scale <- as.numeric(wave_surge$wave_scale)
  inv_softplus <- wave_surge$inv_softplus
  softplus <- wave_surge$softplus

  # The saved functions use base R only. Detaching their source environments
  # prevents either function from retaining the full preprocessing list.
  if (is.function(inv_softplus)) environment(inv_softplus) <- baseenv()
  if (is.function(softplus)) environment(softplus) <- baseenv()

  n <- length(data_dates)
  if (!n || anyNA(data_dates) || is.unsorted(data_dates) ||
      anyDuplicated(data_dates)) {
    stop("wave_surge$date must contain increasing, unique, valid dates.")
  }
  if (!is.matrix(data_y) || !identical(dim(data_y), c(n, 2L)) ||
      !is.matrix(data_y_original) ||
      !identical(dim(data_y_original), c(n, 2L)) ||
      !is.matrix(data_x) || nrow(data_x) != n) {
    stop("Unexpected response or tabular-predictor dimensions.")
  }
  image_dimensions <- dim(data_x_image)
  if (length(image_dimensions) != 4L || image_dimensions[1L] != n ||
      image_dimensions[2L] != 3L ||
      !identical(dimnames(data_x_image)[[2L]], c("psl", "uas", "vas"))) {
    stop("Expected x_image in [case, psl/uas/vas, row, column] order.")
  }
  expected_image_summary_dimensions <- image_dimensions[2:4]
  if (!identical(dim(image_center), expected_image_summary_dimensions) ||
      !identical(dim(image_scale), expected_image_summary_dimensions) ||
      !identical(dimnames(image_center)[[1L]], c("psl", "uas", "vas")) ||
      !identical(dimnames(image_scale)[[1L]], c("psl", "uas", "vas")) ||
      any(!is.finite(image_center)) || any(!is.finite(image_scale)) ||
      any(image_scale <= 0)) {
    stop(
      "Expected valid [psl/uas/vas, row, column] image normalization fields."
    )
  }
  if (length(target_center) != 2L || length(target_scale) != 2L ||
      any(!is.finite(target_center)) || any(!is.finite(target_scale)) ||
      any(target_scale <= 0) || length(wave_scale) != 1L ||
      !is.finite(wave_scale) || wave_scale <= 0 ||
      !is.function(inv_softplus) || !is.function(softplus)) {
    stop("Invalid response transformations or normalization statistics.")
  }
  if (any(!is.finite(data_y)) || any(!is.finite(data_y_original)) ||
      any(!is.finite(data_x))) {
    stop("The frozen tabular predictors and responses must all be finite.")
  }

  transform_probe <- c(1e-4, 0.05, 1, 10)
  if (!isTRUE(all.equal(
    softplus(inv_softplus(transform_probe, wave_scale), wave_scale),
    transform_probe,
    tolerance = 1e-10,
    check.attributes = FALSE
  ))) {
    stop("The saved wave-height transformation functions are inconsistent.")
  }

  test_dates <- as.Date(results$test_dates)
  if (anyNA(test_dates) || anyDuplicated(test_dates)) {
    stop("results$test_dates must be valid and unique.")
  }
  test_rows <- match(test_dates, data_dates)
  if (anyNA(test_rows)) {
    stop("Every saved test date must occur in wave_surge$date.")
  }

  month <- as.integer(format(data_dates, "%m"))
  year <- as.integer(format(data_dates, "%Y"))
  season <- year + as.integer(month >= 10L)
  train_seasons <- as.integer(results$train_seasons)
  if (!length(train_seasons) || anyNA(train_seasons)) {
    stop("results$train_seasons must contain valid winter labels.")
  }
  train_rows <- which(season %in% train_seasons)
  if (!length(train_rows)) {
    stop("No saved preprocessing rows match results$train_seasons.")
  }

  x_test <- data_x[test_rows, , drop = FALSE]
  x_image_test <- data_x_image[test_rows, , , , drop = FALSE]
  observed_test <- data_y_original[test_rows, , drop = FALSE]
  observed_test_model <- data_y[test_rows, , drop = FALSE]
  y_train_original <- data_y_original[train_rows, , drop = FALSE]
  if (any(!is.finite(x_image_test))) {
    stop("The matched test image predictors must all be finite.")
  }
  response_names <- colnames(data_y)
  original_response_names <- colnames(data_y_original)
  if (is.null(response_names)) {
    response_names <- c("softplus_Wave.m", "Surge.m")
  }
  if (is.null(original_response_names)) {
    original_response_names <- c("Wave.m", "Surge.m")
  }

  if (!all(interpreted_features %in% colnames(x_test))) {
    stop(
      "The interpreted lag features are absent from wave_surge$x: ",
      paste(setdiff(interpreted_features, colnames(x_test)), collapse = ", "),
      "."
    )
  }
  if (!isTRUE(all.equal(
    as.matrix(results$observed_test),
    observed_test,
    tolerance = 1e-12,
    check.attributes = FALSE
  ))) {
    stop("The fitting results and preprocessing object have different test data.")
  }
  legacy_statistics <- list(
    wave_transform_scale = wave_scale,
    target_center = target_center,
    target_scale = target_scale
  )
  for (item in names(legacy_statistics)) {
    if (!is.null(results[[item]]) && !isTRUE(all.equal(
      as.numeric(results[[item]]),
      legacy_statistics[[item]],
      tolerance = 0,
      check.attributes = FALSE
    ))) {
      stop("The fitting results and preprocessing object disagree on ", item, ".")
    }
  }

  # Release the complete 6990-case image array before loading the model. Only
  # the matched test images and small normalization fields remain in memory.
  rm(wave_surge, data_x_image, data_x, data_y, data_y_original)
  invisible(gc())

  to_model_response <- function(values) {
    value_dimensions <- dim(values)
    value_dimnames <- dimnames(values)
    transformed <- matrix(values, ncol = 2L)
    transformed[, 1L] <- inv_softplus(transformed[, 1L], wave_scale)
    transformed <- sweep(transformed, 2L, target_center, "-")
    transformed <- sweep(transformed, 2L, target_scale, "/")
    dim(transformed) <- value_dimensions
    dimnames(transformed) <- value_dimnames
    transformed
  }

  model_quantile_to_physical <- function(values, response) {
    transformed <- values * target_scale[response] + target_center[response]
    if (response == 1L) {
      softplus(transformed, wave_scale)
    } else {
      transformed
    }
  }

  if (!isTRUE(all.equal(
    to_model_response(observed_test),
    observed_test_model,
    tolerance = 1e-10,
    check.attributes = FALSE
  ))) {
    stop("The saved response transformations do not reproduce wave_surge$y.")
  }
  rm(observed_test_model)

  array_max_difference <- function(current, saved, name) {
    current <- as.array(current)
    saved <- as.array(saved)
    if (!identical(dim(current), dim(saved))) {
      stop("Saved and reloaded prediction dimensions differ for ", name, ".")
    }
    current_value <- as.numeric(current)
    saved_value <- as.numeric(saved)
    same_na <- is.na(current_value) == is.na(saved_value)
    same_infinite <- is.infinite(current_value) == is.infinite(saved_value)
    both_infinite <- is.infinite(current_value) & is.infinite(saved_value)
    same_infinite_value <- !both_infinite |
      (sign(current_value) == sign(saved_value))
    if (any(!same_na | !same_infinite | !same_infinite_value)) {
      stop("Saved and reloaded predictions differ in non-finite values for ",
           name, ".")
    }
    finite <- is.finite(current_value) & is.finite(saved_value)
    if (any(finite)) {
      max(abs(current_value[finite] - saved_value[finite]))
    } else {
      0
    }
  }

  cat("Loading fitted model on", device, "...\n")
  model <- torch_load(model_file, device = device)
  pred_test <- predict_mst_pmdn(
    model,
    new_inputs = x_test,
    image_inputs = x_image_test,
    device = device
  )

  required_prediction_fields <- c("pi", "mu", "scale_chol", "nu", "alpha")
  if (!all(required_prediction_fields %in% names(results$pred_test)) ||
      !all(required_prediction_fields %in% names(pred_test))) {
    stop("Saved or reloaded predictions lack required MST fields.")
  }
  prediction_difference <- vapply(
    required_prediction_fields,
    function(name) array_max_difference(
      pred_test[[name]]$to(device = "cpu"),
      results$pred_test[[name]],
      name
    ),
    numeric(1)
  )
  if (any(prediction_difference > prediction_tolerance)) {
    stop(
      "Reloaded model predictions differ from results$pred_test; maximum ",
      "absolute differences are ",
      paste(
        names(prediction_difference),
        format(prediction_difference, digits = 4),
        sep = "=",
        collapse = ", "
      ),
      "."
    )
  }

  prediction_dimensions <- as.integer(pred_test$mu$size())
  n_mixtures <- prediction_dimensions[2L]
  n_response <- prediction_dimensions[3L]
  if (n_response != 2L ||
      n_mixtures != as.integer(results$n_mixtures)) {
    stop("The fitted model dimensions disagree with the saved fitting settings.")
  }
  single_component <- n_mixtures == 1L

  physical_threshold <- apply(
    y_train_original,
    2L,
    stats::quantile,
    probs = threshold_probability,
    names = FALSE,
    type = 8
  )
  names(physical_threshold) <- original_response_names
  model_threshold <- as.numeric(to_model_response(matrix(
    physical_threshold,
    nrow = 1L,
    dimnames = list(NULL, original_response_names)
  )))
  names(model_threshold) <- response_names

  joint_event <- mst_functional(
    "joint_exceedance",
    responses = c(1L, 2L),
    threshold = model_threshold,
    direction = c("upper", "upper")
  )
  functional_specifications <- list(
    joint_exceedance = joint_event,
    wave_exceedance = mst_functional(
      "exceedance", responses = 1L, threshold = model_threshold[1L]
    ),
    surge_exceedance = mst_functional(
      "exceedance", responses = 2L, threshold = model_threshold[2L]
    ),
    wave_median = mst_functional(
      "quantile", responses = 1L, prob = 0.5
    ),
    wave_q95 = mst_functional(
      "quantile", responses = 1L, prob = threshold_probability
    ),
    surge_median = mst_functional(
      "quantile", responses = 2L, prob = 0.5
    ),
    surge_q95 = mst_functional(
      "quantile", responses = 2L, prob = threshold_probability
    ),
    correlation = mst_functional(
      "correlation", responses = c(1L, 2L)
    )
  )
  if (single_component) {
    functional_specifications$df <- mst_functional("df")
    functional_specifications$wave_skew_direction <- mst_functional(
      "skew_direction", responses = 1L
    )
    functional_specifications$surge_skew_direction <- mst_functional(
      "skew_direction", responses = 2L
    )
  }

  latent_bank <- latent_draws_mst_pmdn(
    num_samples = num_samples,
    output_dim = n_response,
    dtype = pred_test$mu$dtype,
    device = device,
    seed = xai_seed
  )
  report_tail_resolution <- function(label, object) {
    minimum <- object$diagnostics$min_expected_tail_draws
    low <- object$diagnostics$low_tail_resolution_evaluations
    if (is.null(low)) {
      low <- length(object$diagnostics$low_tail_resolution_rows)
    }
    cat(
      sprintf(
        paste0(
          "Tail resolution [%s]: minimum expected draws=%s; ",
          "low evaluations=%d; threshold=%d.\n"
        ),
        label,
        if (is.finite(minimum)) format(minimum, digits = 5) else "NA",
        as.integer(low),
        min_tail_draws
      )
    )
    invisible(object)
  }
  evaluate_functional <- function(functional) {
    functional_mst_pmdn(
      pred = pred_test,
      functional = functional,
      latent_draws = latent_bank,
      chunk_size = chunk_size,
      device = device,
      response_names = response_names,
      min_tail_draws = min_tail_draws
    )
  }

  cat("Evaluating predictive functionals...\n")
  functional_results <- lapply(
    functional_specifications,
    evaluate_functional
  )
  for (name in names(functional_results)) {
    functional_results[[name]]$data$date <- test_dates
  }

  prediction_summary <- data.frame(
    date = test_dates,
    joint_exceedance_probability =
      functional_results$joint_exceedance$data$value,
    wave_exceedance_probability =
      functional_results$wave_exceedance$data$value,
    surge_exceedance_probability =
      functional_results$surge_exceedance$data$value,
    wave_median_m = model_quantile_to_physical(
      functional_results$wave_median$data$value, 1L
    ),
    wave_q95_m = model_quantile_to_physical(
      functional_results$wave_q95$data$value, 1L
    ),
    surge_median_m = model_quantile_to_physical(
      functional_results$surge_median$data$value, 2L
    ),
    surge_q95_m = model_quantile_to_physical(
      functional_results$surge_q95$data$value, 2L
    ),
    model_scale_correlation = functional_results$correlation$data$value
  )
  if (single_component) {
    prediction_summary$df <- functional_results$df$data$value
    prediction_summary$wave_skew_direction <-
      functional_results$wave_skew_direction$data$value
    prediction_summary$surge_skew_direction <-
      functional_results$surge_skew_direction$data$value
  }

  for (name in names(functional_results)) {
    if (identical(
      functional_results[[name]]$settings$method,
      "monte_carlo"
    )) {
      report_tail_resolution(
        paste("functional", name),
        functional_results[[name]]
      )
    }
  }

  finite_joint <- which(is.finite(
    prediction_summary$joint_exceedance_probability
  ))
  if (!length(finite_joint)) {
    stop("No finite joint exceedance probabilities are available.")
  }
  ranked_cases <- finite_joint[order(
    prediction_summary$joint_exceedance_probability[finite_joint],
    decreasing = TRUE
  )]

  select_evenly <- function(index, n) {
    if (!length(index) || n < 1L) return(integer(0))
    index <- as.integer(index)
    positions <- unique(as.integer(round(seq(
      1L, length(index), length.out = min(as.integer(n), length(index))
    ))))
    index[positions]
  }
  select_risk_stratified_cases <- function(probability, n) {
    finite <- which(is.finite(probability))
    n <- min(as.integer(n), length(finite))
    if (n < 1L) return(integer(0))
    if (n == length(finite)) return(finite)

    nonpositive <- finite[probability[finite] <= 0]
    positive <- finite[probability[finite] > 0]
    strata <- list()
    if (length(nonpositive)) strata$zero <- nonpositive
    if (length(positive)) {
      positive <- positive[order(probability[positive], positive)]
      quartile <- pmin(
        4L,
        as.integer(ceiling(4L * seq_along(positive) / length(positive)))
      )
      positive_strata <- split(positive, quartile)
      names(positive_strata) <- paste0("positive_q", names(positive_strata))
      strata <- c(strata, positive_strata)
    }
    strata <- strata[lengths(strata) > 0L]

    allocation <- rep(n %/% length(strata), length(strata))
    if (n %% length(strata)) {
      allocation[seq_len(n %% length(strata))] <-
        allocation[seq_len(n %% length(strata))] + 1L
    }
    allocation <- pmin(allocation, lengths(strata))
    while (sum(allocation) < n) {
      available <- which(allocation < lengths(strata))
      if (!length(available)) break
      for (i in available) {
        allocation[i] <- allocation[i] + 1L
        if (sum(allocation) == n) break
      }
    }
    sort(unique(unlist(Map(select_evenly, strata, allocation))))
  }

  ice_display_cases <- select_risk_stratified_cases(
    prediction_summary$joint_exceedance_probability,
    n_ice_display_curves
  )
  ice_inputs <- x_test[ice_display_cases, , drop = FALSE]
  ice_image_inputs <- x_image_test[
    ice_display_cases, , , , drop = FALSE
  ]

  feature_metadata <- list(
    wave_transformed_lag1 = list(
      response = 1L,
      label = "Lag-1 significant wave height (m)"
    ),
    surge_lag1 = list(
      response = 2L,
      label = "Lag-1 daily maximum surge (m)"
    )
  )

  feature_to_physical <- function(values, feature) {
    response <- feature_metadata[[feature]]$response
    model_quantile_to_physical(values, response)
  }
  physicalize_ale <- function(object, feature) {
    for (column in c("lower", "upper", "feature_value")) {
      object$data[[column]] <- feature_to_physical(
        object$data[[column]], feature
      )
    }
    object$breaks <- feature_to_physical(object$breaks, feature)
    object$feature$model_name <- object$feature$name
    object$feature$name <- feature_metadata[[feature]]$label
    object$feature$units <- "m"
    object
  }
  physicalize_ice <- function(object, feature) {
    curve_scaled <- object$curves$feature_value
    object$curves$feature_value <- feature_to_physical(curve_scaled, feature)
    object$grid <- feature_to_physical(object$grid, feature)
    object$reference <- feature_to_physical(object$reference, feature)
    object$feature$model_name <- object$feature$name
    object$feature$name <- feature_metadata[[feature]]$label
    object$feature$units <- "m"
    if (!is.null(object$ale)) {
      object$ale <- physicalize_ale(object$ale, feature)
    }
    object
  }

  ale_results <- list()
  ice_results <- list()
  for (feature in interpreted_features) {
    cat("Computing ALE and centred ICE for", feature, "...\n")
    ice_grid <- unique(as.numeric(stats::quantile(
      x_test[, feature],
      probs = seq(0, 1, length.out = n_ice_grid_points),
      names = FALSE,
      type = 7
    )))
    if (length(ice_grid) < 2L) {
      stop("The ICE feature grid must contain at least two distinct values.")
    }
    ale_model_scale <- ale_mst_pmdn(
      model = model,
      inputs = x_test,
      image_inputs = x_image_test,
      feature = feature,
      functional = joint_event,
      n_bins = n_ale_bins,
      decompose = single_component,
      latent_draws = latent_bank,
      chunk_size = chunk_size,
      device = device,
      response_names = response_names,
      min_tail_draws = min_tail_draws
    )
    report_tail_resolution(paste("ALE", feature), ale_model_scale)
    ice_model_scale <- ice_mst_pmdn(
      model = model,
      inputs = ice_inputs,
      image_inputs = ice_image_inputs,
      feature = feature,
      functional = joint_event,
      grid = ice_grid,
      n_curves = nrow(ice_inputs),
      ale = ale_model_scale,
      latent_draws = latent_bank,
      chunk_size = chunk_size,
      device = device,
      response_names = response_names,
      min_tail_draws = min_tail_draws
    )
    report_tail_resolution(paste("ICE", feature), ice_model_scale)
    ale_results[[feature]] <- physicalize_ale(ale_model_scale, feature)
    ice_results[[feature]] <- physicalize_ice(ice_model_scale, feature)
    ice_results[[feature]]$curves$date <-
      test_dates[ice_display_cases[ice_results[[feature]]$curves$case]]
  }

  image_cases <- head(ranked_cases, n_image_contrast_cases)
  occlusion_cases <- head(ranked_cases, n_occlusion_cases)

  # Each psl/uas/vas field was standardized grid cell by grid cell. A zero
  # model-scale reference is therefore the physical training climatology.
  # There are no deterministically derived image channels. The whole-image
  # contrast blends all three jointly; spatial occlusion separates the fields
  # without requiring a rebuild_channels callback.
  reference_images <- array(
    0,
    dim = c(1L, image_dimensions[2:4]),
    dimnames = list(NULL, c("psl", "uas", "vas"), NULL, NULL)
  )

  cat("Computing whole-image climatology contrasts...\n")
  image_contrast <- image_contrast_mst_pmdn(
    model = model,
    inputs = x_test,
    image_inputs = x_image_test,
    reference_images = reference_images,
    functional = joint_event,
    cases = image_cases,
    decompose = single_component,
    latent_draws = latent_bank,
    chunk_size = chunk_size,
    device = device,
    response_names = response_names,
    min_tail_draws = min_tail_draws
  )
  report_tail_resolution("whole-image contrast", image_contrast)
  image_contrast$data$date <- test_dates[image_contrast$data$case]

  cat("Computing field-specific spatial occlusion maps...\n")
  occlusion_channel_groups <- list(
    psl = 1L,
    uas = 2L,
    vas = 3L
  )
  occlusion_seeds <- xai_seed + seq_len(n_occlusion_banks)
  if (anyNA(occlusion_seeds) || any(occlusion_seeds < 0L)) {
    stop("Independent occlusion seeds exceed the supported integer range.")
  }
  image_occlusion_replicates <- lapply(
    seq_len(n_occlusion_banks),
    function(bank_index) {
      bank <- latent_draws_mst_pmdn(
        num_samples = num_samples,
        output_dim = n_response,
        dtype = pred_test$mu$dtype,
        device = device,
        seed = occlusion_seeds[bank_index]
      )
      result <- image_occlusion_mst_pmdn(
        model = model,
        inputs = x_test,
        image_inputs = x_image_test,
        reference_images = reference_images,
        functional = joint_event,
        patch_size = patch_size,
        stride = stride,
        cases = occlusion_cases,
        taper = "cosine",
        channel_groups = occlusion_channel_groups,
        decompose = FALSE,
        latent_draws = bank,
        chunk_size = chunk_size,
        device = device,
        response_names = response_names,
        min_tail_draws = min_tail_draws
      )
      report_tail_resolution(
        paste("field occlusion bank", bank_index),
        result
      )
      result
    }
  )

  # The across-bank range is a Monte Carlo sign-stability diagnostic, not
  # a calibrated confidence interval.
  combine_occlusion_banks <- function(results) {
    reference <- results[[1L]]
    data_key <- with(
      reference$data,
      paste(case, group, patch, sep = ":")
    )
    population_key <- with(
      reference$population,
      paste(group, patch, sep = ":")
    )
    for (result in results[-1L]) {
      candidate_data_key <- with(
        result$data,
        paste(case, group, patch, sep = ":")
      )
      candidate_population_key <- with(
        result$population,
        paste(group, patch, sep = ":")
      )
      if (!identical(candidate_data_key, data_key) ||
          !identical(candidate_population_key, population_key)) {
        stop("Independent occlusion runs returned incompatible patch rows.")
      }
    }

    effect_by_bank <- vapply(
      results,
      function(result) result$data$effect,
      numeric(nrow(reference$data))
    )
    reference$data$effect <- rowMeans(effect_by_bank)
    reference$data$bank_min_effect <- apply(effect_by_bank, 1L, min)
    reference$data$bank_max_effect <- apply(effect_by_bank, 1L, max)
    reference$data$sign_stable <-
      reference$data$bank_min_effect > 0 |
      reference$data$bank_max_effect < 0
    reference$data$plot_effect <- ifelse(
      reference$data$sign_stable,
      reference$data$effect,
      NA_real_
    )

    for (i in seq_len(nrow(reference$population))) {
      block <- reference$data[
        reference$data$group == reference$population$group[i] &
          reference$data$patch == reference$population$patch[i],
        ,
        drop = FALSE
      ]
      reference$population$mean_signed_effect[i] <- mean(block$effect)
      reference$population$mean_absolute_effect[i] <- mean(abs(block$effect))
      reference$population$positive_fraction[i] <- mean(block$effect > 0)
    }
    signed_by_bank <- vapply(
      results,
      function(result) result$population$mean_signed_effect,
      numeric(nrow(reference$population))
    )
    reference$population$bank_min_signed_effect <-
      apply(signed_by_bank, 1L, min)
    reference$population$bank_max_signed_effect <-
      apply(signed_by_bank, 1L, max)
    reference$population$sign_stable <-
      reference$population$bank_min_signed_effect > 0 |
      reference$population$bank_max_signed_effect < 0
    reference$population$plot_mean_signed_effect <- ifelse(
      reference$population$sign_stable,
      reference$population$mean_signed_effect,
      NA_real_
    )
    reference$population$plot_mean_absolute_effect <- ifelse(
      reference$population$sign_stable,
      reference$population$mean_absolute_effect,
      NA_real_
    )
    reference$population$plot_positive_fraction <- ifelse(
      reference$population$sign_stable,
      reference$population$positive_fraction,
      NA_real_
    )

    bank_minima <- vapply(
      results,
      function(result) result$diagnostics$min_expected_tail_draws,
      numeric(1)
    )
    reference$diagnostics$min_expected_tail_draws <-
      min(bank_minima, na.rm = TRUE)
    reference$diagnostics$low_tail_resolution_evaluations <- sum(vapply(
      results,
      function(result) {
        result$diagnostics$low_tail_resolution_evaluations
      },
      numeric(1)
    ))
    reference$diagnostics$independent_bank <- lapply(
      results,
      function(result) result$diagnostics
    )
    reference$settings$n_independent_latent_banks <- length(results)
    reference$settings$latent_bank_seeds <- occlusion_seeds
    reference
  }

  image_occlusion <- combine_occlusion_banks(
    image_occlusion_replicates
  )
  image_occlusion$data$date <- test_dates[image_occlusion$data$case]
  rm(image_occlusion_replicates)

  # Mixture-safe tail accounting is only needed for the optional M > 1 page.
  tail_components <- list()
  if (n_mixtures > 1L) {
    cat("Computing surge tail-component accounting...\n")
    tail_components$surge <- tail_components_mst_pmdn(
      pred = pred_test,
      response = 2L,
      threshold = model_threshold[2L],
      latent_draws = latent_bank,
      chunk_size = chunk_size,
      device = device,
      response_names = response_names,
      min_tail_draws = min_tail_draws
    )
    report_tail_resolution(
      "surge mixture tail components",
      tail_components$surge
    )
    tail_components$surge$data$date <-
      test_dates[tail_components$surge$data$row]
  }

  # Direct parameter-channel attribution compares the most extreme predicted
  # joint-event case with a typical-probability case. This attributes the
  # predictive-distribution difference, not the input features themselves.
  case_decomposition <- NULL
  if (single_component) {
    typical_case <- finite_joint[which.min(abs(
      prediction_summary$joint_exceedance_probability[finite_joint] -
        stats::median(
          prediction_summary$joint_exceedance_probability[finite_joint]
        )
    ))]
    extreme_case <- ranked_cases[1L]
    pred_typical <- predict_mst_pmdn(
      model,
      new_inputs = x_test[typical_case, , drop = FALSE],
      image_inputs = x_image_test[typical_case, , , , drop = FALSE],
      device = device
    )
    pred_extreme <- predict_mst_pmdn(
      model,
      new_inputs = x_test[extreme_case, , drop = FALSE],
      image_inputs = x_image_test[extreme_case, , , , drop = FALSE],
      device = device
    )
    case_decomposition <- decompose_mst_pmdn(
      pred_from = pred_typical,
      pred_to = pred_extreme,
      functional = joint_event,
      latent_draws = latent_bank,
      chunk_size = 1L,
      device = device,
      response_names = response_names,
      min_tail_draws = min_tail_draws
    )
    report_tail_resolution(
      "typical-to-extreme decomposition",
      case_decomposition
    )
  }

  # The plotting methods do not require the latent bank. Release these
  # references before rendering to keep the PDF-only workflow compact.
  compact_object <- function(object) {
    if (is.null(object)) return(NULL)
    object$latent_draws <- NULL
    if (inherits(object, "mst_pmdn_ice") && !is.null(object$ale)) {
      object$ale$latent_draws <- NULL
    }
    object
  }
  functional_results <- lapply(functional_results, compact_object)
  ale_results <- lapply(ale_results, compact_object)
  ice_results <- lapply(ice_results, compact_object)
  image_contrast <- compact_object(image_contrast)
  image_occlusion <- compact_object(image_occlusion)
  tail_components <- lapply(tail_components, compact_object)
  case_decomposition <- compact_object(case_decomposition)

  finite_limit <- function(values, absolute = FALSE, fallback = 1 / num_samples) {
    values <- values[is.finite(values)]
    if (!length(values)) return(fallback)
    limit <- if (absolute) max(abs(values)) else max(values)
    if (!is.finite(limit) || limit <= 0) fallback else limit
  }
  occlusion_plot_limits <- list(
    case_effect = rep(
      c(-1, 1) * finite_limit(
        image_occlusion$data$plot_effect,
        absolute = TRUE
      ),
      length.out = 2L
    ),
    mean_signed_effect = rep(
      c(-1, 1) * finite_limit(
        image_occlusion$population$plot_mean_signed_effect,
        absolute = TRUE
      ),
      length.out = 2L
    )
  )

  pdf_device <- if (capabilities("cairo")) {
    grDevices::cairo_pdf
  } else {
    grDevices::pdf
  }
  pdf_device(figure_file, width = 10, height = 8)
  device_open <- TRUE
  on.exit({
    if (device_open && grDevices::dev.cur() > 1L) grDevices::dev.off()
  }, add = TRUE)

  col_wave <- "#0072B2"
  col_surge <- "#D55E00"
  col_joint <- "#009E73"
  col_dependence <- "#CC79A7"
  channel_labels <- c(
    location = "Location",
    scale = "Scale",
    skewness = "Skewness",
    df = "Degrees of freedom"
  )
  channel_colours <- c(
    location = "#0072B2",
    scale = "#E69F00",
    skewness = "#CC79A7",
    df = "#009E73"
  )
  diverging_palette <- grDevices::colorRampPalette(
    c("#2166AC", "white", "#B2182B")
  )(201L)
  absolute_palette <- grDevices::colorRampPalette(
    c("white", "#FDB863", "#B2182B")
  )(101L)

  expand_plot_range <- function(values, fraction = 0.06) {
    limits <- range(values, finite = TRUE)
    if (length(limits) != 2L || any(!is.finite(limits))) {
      return(c(-1, 1) / num_samples)
    }
    span <- diff(limits)
    if (span <= 0) span <- max(abs(limits), 1 / num_samples)
    limits + c(-1, 1) * fraction * span
  }
  draw_support_rug <- function(values,
                               col = grDevices::adjustcolor(
                                 "grey25", alpha.f = 0.35
                               ),
                               ticksize = 0.03) {
    values <- values[is.finite(values)]
    if (!length(values)) return(invisible(0L))
    limits <- graphics::par("usr")[1:2]
    tolerance <- 100 * .Machine$double.eps * max(1, abs(limits))
    keep <- values >= limits[1L] - tolerance &
      values <= limits[2L] + tolerance
    if (any(keep)) {
      graphics::rug(
        values[keep],
        col = col,
        ticksize = ticksize,
        quiet = TRUE
      )
    }
    invisible(sum(!keep))
  }
  colour_ticks <- function(zlim) {
    ticks <- pretty(zlim, n = 5L)
    ticks[ticks >= zlim[1L] & ticks <= zlim[2L]]
  }
  draw_vertical_colour_key <- function(zlim,
                                       palette,
                                       label,
                                       compact = FALSE) {
    values <- seq(zlim[1L], zlim[2L], length.out = length(palette))
    if (compact) {
      graphics::par(
        mar = rep(0, 4),
        pty = "m",
        xaxs = "i",
        yaxs = "i"
      )
      graphics::par(plt = c(0.32, 0.48, 0.18, 0.82))
    } else {
      graphics::par(
        mar = c(5.2, 0.2, 4.2, 3.7),
        pty = "m",
        xaxs = "i",
        yaxs = "i"
      )
    }
    graphics::image(
      x = c(0, 1),
      y = values,
      z = rbind(values, values),
      col = palette,
      zlim = zlim,
      axes = FALSE,
      xlab = "",
      ylab = "",
      useRaster = FALSE
    )
    graphics::axis(
      4,
      at = colour_ticks(zlim),
      las = 1,
      cex.axis = if (compact) 0.7 else 0.75,
      tcl = -0.25
    )
    graphics::mtext(
      label,
      side = 4,
      line = if (compact) 2.7 else 2.5,
      cex = if (compact) 0.7 else 0.75
    )
    if (compact) {
      graphics::mtext("Shared scale", side = 3, line = 1.1, cex = 0.8)
    }
    graphics::box()
  }
  draw_horizontal_colour_key <- function(zlim, palette, label) {
    values <- seq(zlim[1L], zlim[2L], length.out = length(palette))
    graphics::par(
      mar = c(2.5, 2.4, 0.2, 0.6),
      mgp = c(1.6, 0.45, 0),
      pty = "m",
      xaxs = "i",
      yaxs = "i"
    )
    graphics::image(
      x = values,
      y = c(0, 1),
      z = cbind(values, values),
      col = palette,
      zlim = zlim,
      axes = FALSE,
      xlab = "",
      ylab = "",
      useRaster = FALSE
    )
    graphics::axis(
      1,
      at = colour_ticks(zlim),
      cex.axis = 0.68,
      tcl = -0.25
    )
    graphics::mtext(label, side = 1, line = 1.45, cex = 0.68)
    graphics::box()
  }
  plot_ale_decomposition <- function(object, label, support) {
    active_channels <- object$active_channels
    panel_columns <- c("ale", paste0("ale_", active_channels))
    panel_labels <- c("Total", unname(channel_labels[active_channels]))
    missing_labels <- is.na(panel_labels)
    panel_labels[missing_labels] <- gsub(
      "_", " ", c("total", active_channels)[missing_labels], fixed = TRUE
    )
    common_ylim <- expand_plot_range(c(
      0,
      unlist(object$data[panel_columns], use.names = FALSE)
    ))
    common_xlim <- expand_plot_range(c(
      object$data$feature_value,
      support
    ), fraction = 0.015)

    graphics::layout(matrix(seq_len(6L), nrow = 2L, byrow = TRUE))
    graphics::par(
      mar = c(2.4, 2.9, 2.1, 0.6),
      oma = c(3.2, 3.5, 2.4, 0.4),
      mgp = c(1.8, 0.55, 0),
      cex.axis = 0.78,
      cex.main = 0.86,
      tcl = -0.25
    )
    for (i in seq_along(panel_columns)) {
      graphics::plot(
        object$data$feature_value,
        object$data[[panel_columns[i]]],
        type = "b",
        xlab = "",
        ylab = "",
        main = panel_labels[i],
        xlim = common_xlim,
        ylim = common_ylim
      )
      graphics::abline(h = 0, lty = 2, col = "grey60")
      draw_support_rug(
        support,
        col = grDevices::adjustcolor("grey25", alpha.f = 0.3),
        ticksize = 0.03
      )
    }
    if (length(panel_columns) < 5L) {
      for (i in seq.int(length(panel_columns) + 1L, 5L)) {
        graphics::plot.new()
      }
    }
    graphics::plot.new()
    graphics::text(
      0.5,
      0.58,
      labels = sprintf(
        "Common y-axis\nMaximum |sum-to-total residual|\n%s",
        format(object$diagnostics$max_abs_sum_to_total_residual, digits = 4)
      ),
      cex = 0.9
    )
    graphics::mtext(
      label,
      side = 1,
      outer = TRUE,
      line = 1.4,
      cex = 0.9
    )
    graphics::mtext(
      "Probability effect",
      side = 2,
      outer = TRUE,
      line = 1.7,
      cex = 0.9
    )
    graphics::mtext(
      paste("ALE parameter-channel decomposition:", label),
      side = 3,
      outer = TRUE,
      line = 0.5,
      cex = 1.05
    )
    graphics::layout(matrix(1L, nrow = 1L))
  }

  plot_channel_bar <- function(values,
                               channels,
                               ylab,
                               main,
                               subtitle = NULL) {
    labels <- unname(channel_labels[channels])
    missing_labels <- is.na(labels)
    labels[missing_labels] <- gsub(
      "_", " ", channels[missing_labels], fixed = TRUE
    )
    colours <- unname(channel_colours[channels])
    colours[is.na(colours)] <- "grey60"
    graphics::barplot(
      values,
      names.arg = labels,
      xlab = "",
      ylab = ylab,
      main = "",
      col = colours,
      ylim = expand_plot_range(c(0, values), fraction = 0.08)
    )
    graphics::abline(h = 0, col = "grey40")
    graphics::mtext(main, side = 3, line = 2.15, cex = 1.1, font = 2)
    if (!is.null(subtitle)) {
      graphics::mtext(subtitle, side = 3, line = 0.55, cex = 0.76)
    }
    invisible(values)
  }

  graphics::par(
    mfrow = c(2, 2),
    mar = c(4, 4.2, 2.7, 1.2),
    mgp = c(2.5, 0.7, 0),
    cex.axis = 0.85,
    cex.lab = 0.9,
    cex.main = 0.9
  )
  graphics::plot(
    test_dates,
    prediction_summary$joint_exceedance_probability,
    type = "l",
    col = col_joint,
    lwd = 1.4,
    xlab = "Date",
    ylab = "Probability",
    main = sprintf(
      "Joint exceedance above training %.0fth percentiles",
      100 * threshold_probability
    )
  )
  graphics::grid()

  graphics::plot(
    test_dates,
    observed_test[, 1L],
    type = "p",
    pch = 16,
    cex = 0.45,
    col = grDevices::adjustcolor("black", alpha.f = 0.55),
    xlab = "Date",
    ylab = "Significant wave height (m)",
    main = "One-step predictive wave quantiles",
    ylim = range(
      observed_test[, 1L],
      prediction_summary$wave_median_m,
      prediction_summary$wave_q95_m,
      finite = TRUE
    )
  )
  graphics::lines(
    test_dates, prediction_summary$wave_median_m,
    col = col_wave, lwd = 1.2
  )
  graphics::lines(
    test_dates, prediction_summary$wave_q95_m,
    col = col_wave, lwd = 1.2, lty = 2
  )
  graphics::abline(h = physical_threshold[1L], col = "grey40", lty = 3)
  graphics::grid()
  graphics::legend(
    "topleft",
    legend = c("Observed", "Median", "95th percentile", "Event threshold"),
    col = c("black", col_wave, col_wave, "grey40"),
    pch = c(16, NA, NA, NA),
    lty = c(NA, 1, 2, 3),
    bty = "n",
    cex = 0.75
  )

  graphics::plot(
    test_dates,
    observed_test[, 2L],
    type = "p",
    pch = 16,
    cex = 0.45,
    col = grDevices::adjustcolor("black", alpha.f = 0.55),
    xlab = "Date",
    ylab = "Daily maximum surge (m)",
    main = "One-step predictive surge quantiles",
    ylim = range(
      observed_test[, 2L],
      prediction_summary$surge_median_m,
      prediction_summary$surge_q95_m,
      finite = TRUE
    )
  )
  graphics::lines(
    test_dates, prediction_summary$surge_median_m,
    col = col_surge, lwd = 1.2
  )
  graphics::lines(
    test_dates, prediction_summary$surge_q95_m,
    col = col_surge, lwd = 1.2, lty = 2
  )
  graphics::abline(h = physical_threshold[2L], col = "grey40", lty = 3)
  graphics::grid()

  graphics::plot(
    test_dates,
    prediction_summary$model_scale_correlation,
    type = "l",
    col = col_dependence,
    lwd = 1.3,
    xlab = "Date",
    ylab = "Conditional correlation",
    main = "Predictive dependence on model-response scale",
    ylim = c(-1, 1)
  )
  graphics::abline(h = 0, col = "grey50", lty = 2)
  graphics::grid()

  if (single_component) {
    graphics::par(
      mfrow = c(2, 2),
      mar = c(4, 4.2, 2.7, 1.2),
      mgp = c(2.5, 0.7, 0),
      cex.axis = 0.85,
      cex.lab = 0.9,
      cex.main = 0.9
    )
    graphics::plot(
      test_dates,
      prediction_summary$df,
      type = "l",
      col = col_dependence,
      lwd = 1.3,
      xlab = "Date",
      ylab = expression(nu),
      main = "Conditional degrees of freedom"
    )
    graphics::grid()

    graphics::plot(
      test_dates,
      prediction_summary$wave_skew_direction,
      type = "l",
      lwd = 1.2,
      col = col_wave,
      xlab = "Date",
      ylab = expression(delta),
      main = "Standardized skew direction",
      ylim = range(
        prediction_summary$wave_skew_direction,
        prediction_summary$surge_skew_direction,
        finite = TRUE
      )
    )
    graphics::lines(
      test_dates,
      prediction_summary$surge_skew_direction,
      col = col_surge,
      lwd = 1.2
    )
    graphics::abline(h = 0, col = "grey50", lty = 2)
    graphics::grid()
    graphics::legend(
      "topleft",
      legend = c("Wave coordinate", "Surge coordinate"),
      col = c(col_wave, col_surge),
      lty = 1,
      bty = "n",
      cex = 0.75
    )

    graphics::plot(
      test_dates,
      prediction_summary$wave_exceedance_probability,
      type = "l",
      lwd = 1.2,
      col = col_wave,
      xlab = "Date",
      ylab = "Probability",
      main = "Marginal threshold exceedance",
      ylim = range(
        prediction_summary$wave_exceedance_probability,
        prediction_summary$surge_exceedance_probability,
        finite = TRUE
      )
    )
    graphics::lines(
      test_dates,
      prediction_summary$surge_exceedance_probability,
      col = col_surge,
      lwd = 1.2
    )
    graphics::grid()
    graphics::legend(
      "topleft",
      legend = c("Wave", "Surge"),
      col = c(col_wave, col_surge),
      lty = 1,
      bty = "n",
      cex = 0.75
    )

    graphics::plot(
      prediction_summary$wave_exceedance_probability,
      prediction_summary$surge_exceedance_probability,
      pch = 16,
      cex = 0.6,
      col = grDevices::adjustcolor(col_joint, alpha.f = 0.55),
      xlab = "Wave exceedance probability",
      ylab = "Surge exceedance probability",
      main = "Marginal tail co-variation"
    )
    graphics::grid()
  }

  for (feature in interpreted_features) {
    label <- feature_metadata[[feature]]$label
    support <- feature_to_physical(x_test[, feature], feature)
    support_xlim <- expand_plot_range(c(
      support,
      ale_results[[feature]]$data$feature_value,
      ice_results[[feature]]$grid
    ), fraction = 0.015)
    ice_display <- ice_results[[feature]]

    graphics::par(
      mfrow = c(1, 2),
      mar = c(3.2, 4.3, 2.8, 1),
      oma = c(2.3, 0, 0, 0),
      mgp = c(2.6, 0.7, 0),
      cex.axis = 0.85,
      cex.lab = 0.9,
      cex.main = 0.95
    )
    plot(
      ale_results[[feature]],
      type = "total",
      xlab = "",
      ylab = "ALE in joint-event probability",
      main = "Accumulated local effect",
      xlim = support_xlim
    )
    draw_support_rug(
      support,
      col = grDevices::adjustcolor("grey25", alpha.f = 0.35),
      ticksize = 0.03
    )
    plot(
      ice_display,
      xlab = "",
      ylab = "Centred joint-event probability",
      main = "Centred ICE",
      xlim = support_xlim
    )
    draw_support_rug(
      support,
      col = grDevices::adjustcolor("grey25", alpha.f = 0.35),
      ticksize = 0.03
    )
    graphics::legend(
      "topleft",
      legend = c(
        sprintf("ICE (%d risk-stratified cases)", length(ice_display$cases)),
        "Population ALE"
      ),
      col = c(
        grDevices::adjustcolor("grey35", alpha.f = 0.45),
        "black"
      ),
      lwd = c(1, 3),
      bty = "n",
      cex = 0.75
    )
    graphics::mtext(
      label,
      side = 1,
      outer = TRUE,
      line = 0.8,
      cex = 0.92
    )

    if (length(ale_results[[feature]]$active_channels)) {
      plot_ale_decomposition(ale_results[[feature]], label, support)
    }
  }

  graphics::par(
    mfrow = c(1, 2),
    mar = c(4.2, 4.3, 2.8, 1),
    mgp = c(2.6, 0.7, 0),
    cex.axis = 0.85,
    cex.lab = 0.88,
    cex.main = 0.92
  )
  graphics::hist(
    image_contrast$data$contrast,
    xlab = "Joint probability: actual fields minus climatological fields",
    ylab = "Frequency",
    main = sprintf(
      "Whole-image contrasts: %d highest-risk cases",
      nrow(image_contrast$data)
    ),
    col = "grey75",
    border = "white"
  )
  graphics::abline(v = 0, lty = 2, col = "grey50")
  contrast_by_date <- image_contrast$data[
    order(image_contrast$data$date),
    ,
    drop = FALSE
  ]
  graphics::plot(
    contrast_by_date$date,
    contrast_by_date$contrast,
    type = "h",
    xlab = "Date",
    ylab = "Joint probability: actual minus climatological fields",
    main = "Selected high-risk dates",
    col = col_joint,
    lwd = 2
  )
  graphics::points(
    contrast_by_date$date,
    contrast_by_date$contrast,
    pch = 16,
    cex = 0.55,
    col = col_joint
  )
  graphics::abline(h = 0, lty = 2, col = "grey50")
  graphics::grid()
  if (length(image_contrast$active_channels)) {
    active_channels <- image_contrast$active_channels
    channel_values <- vapply(
      active_channels,
      function(channel) {
        image_contrast$data[[paste0("channel_", channel)]][1L]
      },
      numeric(1)
    )
    graphics::par(
      mfrow = c(1, 1),
      mar = c(4.5, 4.5, 4.2, 1),
      mgp = c(2.6, 0.7, 0),
      cex.axis = 0.9,
      cex.lab = 0.9
    )
    plot_channel_bar(
      channel_values,
      active_channels,
      "Contribution to joint-event probability",
      paste(
        "Whole-image parameter channels:",
        format(image_contrast$data$date[1L])
      ),
      sprintf(
        "Total contrast: %s; sum-to-total residual: %s",
        format(image_contrast$data$contrast[1L], digits = 4),
        format(image_contrast$data$sum_to_total_residual[1L], digits = 3)
      )
    )
  }

  field_labels <- c(
    psl = "Sea-level pressure",
    uas = "Eastward wind",
    vas = "Northward wind"
  )
  graphics::layout(
    matrix(c(1L, 2L, 3L, 4L, 4L, 4L), nrow = 2L, byrow = TRUE),
    heights = c(4, 0.85)
  )
  population_map_mar <- c(3.1, 4, 2.4, 0.6)
  graphics::par(
    mar = population_map_mar,
    oma = c(2.4, 0, 2.2, 0),
    mgp = c(2.1, 0.55, 0),
    cex.axis = 0.78,
    cex.lab = 0.8,
    cex.main = 0.82,
    pty = "s"
  )
  for (group_index in seq_along(occlusion_channel_groups)) {
    group_name <- names(occlusion_channel_groups)[group_index]
    graphics::par(mar = population_map_mar, pty = "s")
    plot(
      image_occlusion,
      group = group_name,
      statistic = "plot_mean_signed_effect",
      xlab = "Array column",
      ylab = if (group_index == 1L) "Array row" else "",
      main = field_labels[group_name],
      col = diverging_palette,
      zlim = occlusion_plot_limits$mean_signed_effect,
      useRaster = FALSE
    )
  }
  draw_horizontal_colour_key(
    occlusion_plot_limits$mean_signed_effect,
    diverging_palette,
    "Mean signed probability effect"
  )
  graphics::mtext(
    "Field-specific population occlusion: selected high-risk cases",
    side = 3,
    outer = TRUE,
    line = 0.5,
    cex = 1.05
  )
  graphics::mtext(
    sprintf(
      paste0(
        "%d independent latent banks; patches whose bank range covers ",
        "zero are masked. Cosine mask mean %.3f; %d x %d patch; ",
        "stride %d x %d."
      ),
      n_occlusion_banks,
      unique(image_occlusion$patches$mask_mean),
      patch_size[1L],
      patch_size[2L],
      stride[1L],
      stride[2L]
    ),
    side = 1,
    outer = TRUE,
    line = 0.65,
    cex = 0.72
  )
  graphics::layout(matrix(1L, nrow = 1L))

  for (group_name in names(occlusion_channel_groups)) {
    graphics::layout(matrix(seq_len(6L), nrow = 2L, byrow = TRUE))
    individual_map_mar <- c(3.2, 4, 2.3, 0.6)
    graphics::par(
      oma = c(2.8, 0, 2.2, 0),
      mgp = c(2.1, 0.55, 0),
      cex.axis = 0.76,
      cex.lab = 0.8,
      cex.main = 0.82
    )
    for (i in seq_along(occlusion_cases)) {
      case <- occlusion_cases[i]
      panel_row <- if (i <= 3L) 1L else 2L
      panel_column <- if (i <= 3L) i else i - 3L
      graphics::par(
        mar = individual_map_mar,
        pty = "s"
      )
      plot(
        image_occlusion,
        case = case,
        group = group_name,
        statistic = "plot_effect",
        xlab = if (panel_row == 2L) "Array column" else "",
        ylab = if (panel_column == 1L) "Array row" else "",
        main = format(test_dates[case]),
        col = diverging_palette,
        zlim = occlusion_plot_limits$case_effect,
        useRaster = FALSE
      )
    }
    draw_vertical_colour_key(
      occlusion_plot_limits$case_effect,
      diverging_palette,
      "Probability effect (original - occluded)",
      compact = TRUE
    )
    graphics::mtext(
      paste(
        "Individual",
        field_labels[group_name],
        "occlusion maps: highest-risk cases"
      ),
      side = 3,
      outer = TRUE,
      line = 0.5,
      cex = 1.05
    )
    graphics::mtext(
      paste(
        "All fields and dates share one scale; sign-unstable patches are",
        "masked. Axes follow the saved array orientation."
      ),
      side = 1,
      outer = TRUE,
      line = 0.5,
      cex = 0.75
    )
    graphics::layout(matrix(1L, nrow = 1L))
  }

  if (!is.null(case_decomposition)) {
    active_channels <- case_decomposition$active_channels
    channel_values <- vapply(
      active_channels,
      function(channel) {
        case_decomposition$data[[paste0("channel_", channel)]][1L]
      },
      numeric(1)
    )
    graphics::par(
      mfrow = c(1, 1),
      mar = c(4.5, 4.5, 4.2, 1),
      mgp = c(2.6, 0.7, 0),
      cex.axis = 0.9,
      cex.lab = 0.9
    )
    plot_channel_bar(
      channel_values,
      active_channels,
      "Contribution to joint-event probability",
      "Typical-to-extreme predictive-state decomposition",
      sprintf(
        "Total contrast: %s; sum-to-total residual: %s",
        format(case_decomposition$data$total[1L], digits = 4),
        format(
          case_decomposition$data$sum_to_total_residual[1L],
          digits = 3
        )
      )
    )
  }

  if (n_mixtures > 1L) {
    extreme_case <- ranked_cases[1L]
    graphics::par(mfrow = c(1, 1))
    plot(
      tail_components$surge,
      row = extreme_case,
      main = c(
        "Mixture weight",
        "Surge-tail probability",
        "Weighted contribution"
      )
    )
  }

  grDevices::dev.off()
  device_open <- FALSE

  cat("Saved XAI figures:", figure_file, "\n")
  cat(
    "Physical training thresholds:",
    paste(
      names(physical_threshold),
      format(physical_threshold, digits = 4),
      sep = "=",
      collapse = ", "
    ),
    "\n"
  )
  cat(
    "Maximum saved/reloaded prediction difference:",
    format(max(prediction_difference), digits = 4),
    "\n"
  )

  invisible(NULL)
}

main()

################################################################################
