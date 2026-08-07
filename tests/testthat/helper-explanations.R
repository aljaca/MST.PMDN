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

distribution_explanation_test_model <- torch::nn_module(
  initialize = function(n_mixtures = 1L, image_channel = 1L) {
    self$n_mixtures <- as.integer(n_mixtures)
    self$image_channel <- as.integer(image_channel)
  },
  forward = function(x, image_input = NULL) {
    B <- x$size(1)
    state <- x[, 1L]
    if (!is.null(image_input)) {
      image_value <- image_input[, self$image_channel, , ]$
        reshape(c(B, -1L))$mean(dim = 2L)
      state <- state + image_value
    }

    component_index <- seq_len(self$n_mixtures)
    if (self$n_mixtures == 1L) {
      pi <- torch::torch_ones(
        c(B, 1L), dtype = x$dtype, device = x$device
      )
    } else {
      logits <- torch::torch_stack(lapply(component_index, function(g) {
        (g - (self$n_mixtures + 1) / 2) * 0.4 * state
      }), dim = 2L)
      pi <- torch::nnf_softmax(logits, dim = 2L)
    }

    mu <- torch::torch_stack(lapply(component_index, function(g) {
      torch::torch_stack(list(
        state + 0.35 * (g - 1L),
        -0.55 * state + 0.2 * (g - 1L)
      ), dim = 2L)
    }), dim = 2L)

    scale_chol <- torch::torch_stack(lapply(component_index, function(g) {
      diagonal_1 <- torch::torch_exp(0.08 * state + 0.03 * g)
      diagonal_2 <- torch::torch_exp(-0.05 * state + 0.02 * g)
      zero <- torch::torch_zeros_like(state)
      torch::torch_stack(list(
        torch::torch_stack(list(diagonal_1, zero), dim = 2L),
        torch::torch_stack(
          list(0.08 * torch::torch_tanh(state), diagonal_2), dim = 2L
        )
      ), dim = 2L)
    }), dim = 2L)

    alpha <- torch::torch_stack(lapply(component_index, function(g) {
      torch::torch_stack(list(
        0.35 + 0.15 * state + 0.05 * g,
        -0.25 + 0.1 * state - 0.03 * g
      ), dim = 2L)
    }), dim = 2L)
    nu <- torch::torch_stack(lapply(component_index, function(g) {
      5.5 + g + torch::nnf_softplus(0.3 * state)
    }), dim = 2L)

    list(
      pi = pi,
      mu = mu,
      scale_chol = scale_chol,
      nu = nu,
      alpha = alpha,
      skew_none = FALSE
    )
  }
)
