explanation_test_model <- torch::nn_module(
  initialize = function(slope = 1, image_interaction = FALSE,
                        image_channel = 1L) {
    self$slope <- slope
    self$image_interaction <- image_interaction
    self$image_channel <- as.integer(image_channel)
  },
  forward = function(x, image_input = NULL) {
    B <- x$size(1)
    location <- self$slope * x[, 1L]
    if (self$image_interaction) {
      if (is.null(image_input)) stop("image_input is required")
      image_value <- image_input[, self$image_channel, 1L, 1L]
      location <- location * image_value
    } else if (!is.null(image_input)) {
      location <- location + image_input[, self$image_channel, 1L, 1L]
    }
    list(
      pi = torch::torch_ones(
        c(B, 1L), dtype = x$dtype, device = x$device
      ),
      mu = location$view(c(B, 1L, 1L)),
      scale_chol = torch::torch_ones(
        c(B, 1L, 1L, 1L), dtype = x$dtype, device = x$device
      ),
      nu = torch::torch_full(
        c(B, 1L), Inf, dtype = x$dtype, device = x$device
      ),
      alpha = torch::torch_zeros(
        c(B, 1L, 1L), dtype = x$dtype, device = x$device
      ),
      skew_none = TRUE
    )
  }
)
