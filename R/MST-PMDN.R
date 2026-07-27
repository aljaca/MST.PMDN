################################################################################
# Deep Multivariate skew t-Parsimonious Mixture Density Network (MST-PMDN)     #
# Alex J. Cannon <alex.cannon@ec.gc.ca>                                        #
################################################################################

library(torch)

if (getRversion() >= "2.15.1") {
  utils::globalVariables("self")
}

# -----------------
# Utility functions
# -----------------

sample_gamma <- function(shape, scale = 1, device = "cpu") {
  # Gamma scaling for Student-t tails
  if (!inherits(shape, "torch_tensor")) {
    shape <- torch_tensor(shape, device = device, dtype = torch_float())
  } else {
    shape <- shape$to(device = device)
  }
  if (!inherits(scale, "torch_tensor")) {
    scale <- torch_tensor(scale, device = device, dtype = torch_float())
  } else {
    scale <- scale$to(device = device)
  }
  shape_cpu <- as.numeric(shape$to(device = "cpu"))
  scale_cpu <- as.numeric(scale$to(device = "cpu"))
  # Sample using rgamma(shape, scale) where rate = 1/scale
  samples_r <- mapply(rgamma, n = 1, shape = shape_cpu, rate = 1 / scale_cpu)
  out <- torch_tensor(samples_r, dtype = torch_float())$
    reshape(shape$size())$
    to(device = device)
  return(out)
}

validate_num_samples <- function(num_samples) {
  if (!is.numeric(num_samples) || length(num_samples) != 1 ||
      !is.finite(num_samples) || num_samples < 1 ||
      num_samples != floor(num_samples)) {
    stop("num_samples must be a positive integer.")
  }
  as.integer(num_samples)
}

build_orthogonal_matrix <- function(params, dim) {
  # Helper for building an orthogonal orientation matrix
  dev <- params$device
  batch_size <- params$size(1)
  X <- torch_zeros(batch_size, dim, dim, device = dev)
  indices_orig <- torch_triu_indices(dim, dim, offset = 1,
                                     dtype = torch_long(), device = dev)
  row_indices_1d <- indices_orig[1, ]$to(dtype = torch_long())
  col_indices_1d <- indices_orig[2, ]$to(dtype = torch_long())
  batch_vals <- torch_arange(1, batch_size, device = dev, dtype = torch_long())
  # Expand batch, row, and column indices for broadcasting
  num_triu_elements <- indices_orig$size(2)
  batch_indices <- batch_vals$unsqueeze(2)$expand(c(batch_size,
                     num_triu_elements))
  row_indices <- row_indices_1d$unsqueeze(1)$expand(c(batch_size,
                   num_triu_elements))
  col_indices <- col_indices_1d$unsqueeze(1)$expand(c(batch_size,
                   num_triu_elements))
  X$index_put_(list(batch_indices, row_indices, col_indices), params)
  X <- X - X$transpose(2, 3)
  Q <- torch_matrix_exp(X)
  Q <- linalg_qr(Q)[[1]]
  Q
}

init_mu_kmeans <- function(model, outputs_train, n_mixtures, constant_attr,
                           device = "cpu") {
  # Initialize mu with centroids from k-means clustering
  km    <- kmeans(as.matrix(outputs_train), centers = n_mixtures, nstart = 20)
  cent  <- torch_tensor(km$centers, dtype = torch_float(), device = device)
  if (grepl("m", constant_attr)) {
    # mu is a parameter
    with_no_grad({
      model$mu$copy_(cent)
    })
  } else {
    # mu comes from bias of fc_mu
    with_no_grad({
      model$fc_mu$bias$copy_(cent$reshape(c(-1)))
      model$fc_mu$g$zero_()
    })
  }
}

# --------------------------------
# Differentiable Student-t CDF
# --------------------------------

.coerce_t_cdf_inputs <- function(z, nu) {
  if (!inherits(z, "torch_tensor")) {
    z <- torch_tensor(z, dtype = torch_float())
  }
  if (!inherits(nu, "torch_tensor")) {
    nu <- torch_tensor(nu, dtype = z$dtype, device = z$device)
  } else {
    nu <- nu$to(dtype = z$dtype, device = z$device)
  }
  list(z = z, nu = nu)
}

.hill_t_transform <- function(z, nu) {
  # Hill's normalizing transformation, written without sign() or the
  # removable singularity at z = 0.
  dtype <- z$dtype
  device <- z$device
  one <- torch_tensor(1, dtype = dtype, device = device)
  a <- nu - 0.5 * one
  u <- z$pow(2) / nu
  u_safe <- torch_where(u == 0, one, u)
  log1p_u <- torch_log1p(u)
  ratio_direct <- log1p_u / u_safe
  ratio_series <- one - u / 2 + u$pow(2) / 3 - u$pow(3) / 4 +
                  u$pow(4) / 5
  log1p_ratio <- torch_where(u < 1e-4, ratio_series, ratio_direct)

  # Brophy's algebraic form of Hill's three-term expansion. The signed
  # leading term q is smooth through zero because log1p(u) / u is evaluated
  # by its series there.
  r <- a * log1p_u
  b <- 48 * a$pow(2)
  polynomial <- ((0.4 * r + 3.3 * one) * r + 24 * one) * r +
                85.5 * one
  correction <- one + (r + 3 * one) / b -
                polynomial / (b * (0.8 * r$pow(2) + 100 * one + b))
  q <- z * torch_sqrt((a / nu) * log1p_ratio)

  q * correction
}

.log_normal_cdf <- function(z) {
  # log(Phi(z)) without cancellation from 0.5 * (1 + erf(z / sqrt(2))).
  # erfc is accurate through the range relevant to float32. The asymptotic
  # branch keeps the result and its gradient finite beyond that range.
  dtype <- z$dtype
  device <- z$device
  one <- torch_tensor(1, dtype = dtype, device = device)
  half <- 0.5 * one
  sqrt_two <- torch_tensor(sqrt(2), dtype = dtype, device = device)

  # Clamp only the inactive erfc branch so it cannot underflow before
  # torch_where selects the asymptotic result.
  z_erfc <- torch_clamp(z, min = -10)
  log_erfc <- torch_log(half) +
              torch_log(torch_erfc(-z_erfc / sqrt_two))

  abs_z <- torch_clamp(torch_abs(z), min = 1)
  inv_z2 <- abs_z$pow(-2)
  mills <- one - inv_z2 + 3 * inv_z2$pow(2) -
           15 * inv_z2$pow(3) + 105 * inv_z2$pow(4)
  log_asymptotic <- -0.5 * z$pow(2) - torch_log(abs_z) -
                    0.5 * log(2 * pi) +
                    torch_log(mills)

  torch_where(z < -10, log_asymptotic, log_erfc)
}

.log_t1_cdf <- function(z) {
  # The direct Cauchy expression loses precision for large negative z.
  dtype <- z$dtype
  device <- z$device
  one <- torch_tensor(1, dtype = dtype, device = device)
  half <- 0.5 * one
  pi_t <- torch_tensor(pi, dtype = dtype, device = device)
  negative_tail <- z < -1
  z_tail <- torch_where(negative_tail, z, -one)
  z_direct <- torch_clamp(z, min = -1)
  log_tail <- torch_log(torch_atan(-one / z_tail) / pi_t)
  log_direct <- torch_log(half + torch_atan(z_direct) / pi_t)
  torch_where(negative_tail, log_tail, log_direct)
}

.log_t2_cdf <- function(z) {
  # Use an algebraically equivalent lower-tail expression when z is negative
  # to avoid subtracting nearly equal values.
  dtype <- z$dtype
  device <- z$device
  one <- torch_tensor(1, dtype = dtype, device = device)
  two <- 2 * one
  half <- 0.5 * one
  negative_tail <- z < -1
  z_tail <- torch_where(negative_tail, z, -one)
  root_tail <- torch_sqrt(two + z_tail$pow(2))
  log_tail <- -torch_log(root_tail) - torch_log(root_tail - z_tail)
  z_direct <- torch_clamp(z, min = -1)
  root_direct <- torch_sqrt(two + z_direct$pow(2))
  log_direct <- torch_log(half + z_direct / (two * root_direct))
  torch_where(negative_tail, log_tail, log_direct)
}

log_pt <- function(z, nu) {
  args <- .coerce_t_cdf_inputs(z, nu)
  z <- args$z
  nu <- args$nu
  dtype <- z$dtype
  device <- z$device
  one <- torch_tensor(1, dtype = dtype, device = device)
  two <- 2 * one
  normal <- nu == Inf
  nu_safe <- torch_where(normal, 3 * one, nu)

  out <- .log_normal_cdf(.hill_t_transform(z, nu_safe))
  out <- torch_where(nu_safe == two, .log_t2_cdf(z), out)
  out <- torch_where(nu_safe == one, .log_t1_cdf(z), out)
  torch_where(normal, .log_normal_cdf(z), out)
}

t_cdf <- function(z, nu) {
  torch_exp(log_pt(z, nu))
}

.log_multivariate_t_normalizer <- function(nu, d) {
  # Evaluate
  # log Gamma((nu + d) / 2) - log Gamma(nu / 2)
  # - (d / 2) log(nu * pi)
  # without subtracting large, nearly equal float32 values. For even d,
  # Gamma recurrence removes the gamma functions entirely. Odd-dimensional
  # ratios are small tensors, so evaluate them in float64 before casting back.
  if (d %% 2 == 0) {
    m <- as.integer(d / 2)
    out <- 0 * nu - m * log(2 * pi)
    if (m > 1) {
      for (j in seq_len(m - 1)) {
        out <- out + torch_log1p(2 * j / nu)
      }
    }
    return(out)
  }

  dtype <- nu$dtype
  nu_double <- nu$to(dtype = torch_double())
  out <- torch_lgamma((nu_double + d) / 2) -
         torch_lgamma(nu_double / 2) -
         (d / 2) * (torch_log(nu_double) + log(pi))
  out$to(dtype = dtype)
}

# ------------------------------
# Weight-normalized linear layer
# ------------------------------

weight_norm_linear <- nn_module(
  "weight_norm_linear",
  initialize = function(in_features, out_features, bias = TRUE) {
    self$in_features <- in_features
    self$out_features <- out_features
    # Parameters V and g
    self$V <- nn_parameter(torch_randn(out_features, in_features) /
                             sqrt(in_features))
    self$g <- nn_parameter(torch_ones(out_features))
    if (bias) {
      self$bias <- nn_parameter(torch_zeros(out_features))
    } else {
      self$register_parameter("bias", NULL)
    }
  },
  forward = function(input) {
    # Normalize V along input dimension
    V_norm <- self$V / torch_norm(self$V, dim = 2, keepdim = TRUE)
    # Apply g scaling factor
    W <- self$g$unsqueeze(2) * V_norm
    # Standard linear operation
    output <- input$matmul(W$t())
    if (!is.null(self$bias)) {
      output <- output + self$bias
    }
    output
  }
)

init_weight_norm <- function(module) {
  if (inherits(module, "weight_norm_linear")) {
    # Initialize V using He initialization
    nn_init_kaiming_normal_(module$V, mode = "fan_out")
    # Initialize g to match the original scale
    with_no_grad({
      norm_v <- torch_norm(module$V, dim = 2)
      module$g$copy_(norm_v)
    })
  }
}

init_distribution_heads <- function(model) {
  # Apply specialized initialization after the generic weight-normalized
  # initialization so learned nu and alpha begin input-independent.
  with_no_grad({
    for (head_name in c("fc_nu", "fc_nu_partial")) {
      head <- model[[head_name]]
      if (!is.null(head)) {
        head$V$normal_(0, 0.02)
        head$g$zero_()
        head$bias$zero_()
      }
    }
    if (!is.null(model$fc_alpha)) {
      model$fc_alpha$V$normal_(0, 0.02)
      model$fc_alpha$g$zero_()
      model$fc_alpha$bias$zero_()
    }
  })
}

derive_checkpoint_path <- function(path, suffix) {
  derived <- sub("\\.pt$", paste0(suffix, ".pt"), path)
  if (identical(derived, path)) {
    derived <- paste0(path, suffix)
  }
  derived
}

capture_training_rng_state <- function() {
  list(
    r = if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
    } else {
      NULL
    },
    torch = torch_get_rng_state(),
    cuda = if (cuda_is_available()) cuda_get_rng_state() else NULL
  )
}

restore_training_rng_state <- function(state) {
  if (is.null(state)) {
    return(invisible(FALSE))
  }
  if (!is.null(state$r)) {
    assign(".Random.seed", state$r, envir = .GlobalEnv)
  }
  if (!is.null(state$torch)) {
    torch_set_rng_state(state$torch)
  }
  if (!is.null(state$cuda) && cuda_is_available()) {
    cuda_set_rng_state(state$cuda)
  }
  invisible(TRUE)
}

# -----------------------------------------
# Skew t-distribution PMDN model definition
# -----------------------------------------

define_mst_pmdn <- function(
  input_dim, output_dim, hidden_dim, n_mixtures,
  constraint = "VVVNN",
  constant_attr = "",
  activation = nn_relu,
  drop_hidden = 0.,
  image_module = NULL,
  tabular_module = NULL,
  fusion_module = NULL,
  fixed_nu = NULL,
  range_nu = c(3., 50.),     # clamp learned nu range
  max_alpha = 2.5,          # alpha = [-max_alpha, max_alpha]
  min_vol_shape = 1e-2,     # clamps on L_val and A_diag
  min_mix_weight = 1e-4,    # clamp on min component weight
  jitter = 1e-6             # diagonal ridge for chol
) {
  nn_module(
    get_module_output_dim = function(module, fallback_input_dim = NULL,
                                     module_name = "module") {
      # Attempt to infer output dimension
      if (!is.null(module$output_dim)) {
        return(module$output_dim)
      } else if (!is.null(module$out_features)) {
        return(module$out_features)
      } else if (!is.null(module$out_channels)) {
        return(module$out_channels)
      } else if (!is.null(module$out_dim)) {
        return(module$out_dim)
      } else if (inherits(module, "nn_sequential")) {
        # For sequential, try to get dimension from last layer
        last_layer <- module[[length(module)]]
        if (!is.null(last_layer$out_features)) {
          return(last_layer$out_features)
        } else if (!is.null(last_layer$out_channels)) {
          return(last_layer$out_channels)
        }
      }
      if (is.null(fallback_input_dim)) {
        stop(paste0("Cannot infer output dimension from ", module_name,
                    ". Please ensure the module has one of these attributes: ",
                    "output_dim, out_features, out_channels, out_dim, or use a wrapper."))
      }
      # Use fallback with warning
      warning(paste0("Could not infer output dimension from ", module_name,
                     ". Using fallback dimension: ", fallback_input_dim))
      return(fallback_input_dim)
    },
    initialize = function() {
      # Store user arguments
      self$image_module    <- image_module
      self$tabular_module  <- tabular_module
      self$fusion_module   <- fusion_module
      self$hidden_dims     <- as.integer(hidden_dim)
      self$n_mixtures      <- n_mixtures
      self$output_dim      <- output_dim
      self$constraint      <- constraint
      self$constant_attr   <- constant_attr
      if (!is.numeric(range_nu) || length(range_nu) != 2 ||
          any(!is.finite(range_nu)) || range_nu[1] <= 0 ||
          range_nu[2] <= range_nu[1]) {
        stop("range_nu must be two increasing, positive finite values.")
      }
      self$min_nu          <- range_nu[1]
      self$max_nu          <- range_nu[2]
      self$max_alpha       <- max_alpha
      self$min_vol_shape   <- min_vol_shape
      self$min_mix_weight  <- min_mix_weight
      self$jitter          <- jitter
      if (is.null(self$fusion_module)) {
        # Infer output dimensions from modules
        # Determine tabular features dimension
        if (is.null(self$tabular_module)) {
          tabular_features_dim <- input_dim
        } else {
          # Try to infer output dimension from the tabular module
          tabular_features_dim <- self$get_module_output_dim(
            self$tabular_module,
            fallback_input_dim = input_dim,
            module_name = "tabular_module"
          )
        }
        # Calculate total input dimensions after feature extraction
        total_input_dim <- tabular_features_dim
        if (!is.null(self$image_module)) {
          # Try to infer output dimension from the image module
          image_out_dim <- self$get_module_output_dim(
            self$image_module,
            fallback_input_dim = NULL,
            module_name = "image_module"
          )
          total_input_dim <- total_input_dim + image_out_dim
        }
        # Build hidden MLP
        if (is.function(activation)) {
          act_funcs <- rep(list(activation), length(self$hidden_dims))
        } else if (is.list(activation) && length(activation) ==
                     length(self$hidden_dims)) {
          act_funcs <- activation
        } else {
          stop("activation must match number of hidden layers.")
        }
        layers <- list()
        current_dim <- total_input_dim
        n_hidden_layers <- length(self$hidden_dims)
        for (i in seq_len(n_hidden_layers)) {
          next_dim <- self$hidden_dims[i]
          layers[[length(layers) + 1]] <- nn_linear(current_dim, next_dim)
          # Add batch norm and activation except on the final (output) layer
          if (i < n_hidden_layers) {
            layers[[length(layers) + 1]] <- nn_batch_norm1d(next_dim)
            layers[[length(layers) + 1]] <- act_funcs[[i]]()
            layers[[length(layers) + 1]] <- nn_dropout(p = drop_hidden)
          }
          current_dim <- next_dim
        }
        self$hidden <- nn_sequential(!!!layers)
        self$final_hidden_dim <- current_dim
      } else {
        if (length(self$hidden_dims) > 0) {
          fallback_dim <- self$hidden_dims[length(self$hidden_dims)]
        } else {
          fallback_dim <- NULL
        }
        self$hidden <- NULL
        if (drop_hidden > 0) {
          self$fusion_dropout <- nn_dropout(p = drop_hidden)
        } else {
          self$fusion_dropout <- NULL
        }
        self$final_hidden_dim <- self$get_module_output_dim(
          self$fusion_module,
          fallback_input_dim = fallback_dim,
          module_name = "fusion_module"
        )
      }
      # --------------------
      # Mixture weights (pi)
      # --------------------
      if (grepl("x", constant_attr)) {
        self$pi <- nn_parameter(torch_ones(n_mixtures) / n_mixtures)
      } else {
        self$fc_pi <- weight_norm_linear(self$final_hidden_dim, n_mixtures)
      }
      # ----------
      # Locations (mu)
      # ----------
      if (grepl("m", constant_attr)) {
        self$mu <- nn_parameter(torch_randn(n_mixtures, output_dim))
      } else {
        self$fc_mu <- weight_norm_linear(self$final_hidden_dim,
                                         n_mixtures * output_dim)
      }
      # ----------
      # Volume (L)
      # ----------
      self$volume_shared <- substr(constraint, 1, 1) == "E"
      volume_size <- if (self$volume_shared) 1 else n_mixtures
      if (grepl("L", constant_attr)) {
        self$L_param <- nn_parameter(torch_zeros(volume_size))
      } else {
        self$fc_L <- weight_norm_linear(self$final_hidden_dim, volume_size)
      }
      # ---------
      # Shape (A)
      # ---------
      self$shape_shared    <- substr(constraint, 2, 2) == "E"
      self$shape_identity  <- substr(constraint, 2, 2) == "I"
      if (!self$shape_identity) {
        if (grepl("A", constant_attr)) {
          if (self$shape_shared) {
            self$A_param <- nn_parameter(0.1 * torch_randn(output_dim))
          } else {
            self$A_param <- nn_parameter(0.1 * torch_randn(n_mixtures,
                                                           output_dim))
          }
        } else {
          shape_size <- if (self$shape_shared) output_dim else (n_mixtures *
                                                                  output_dim)
          self$fc_A <- weight_norm_linear(self$final_hidden_dim, shape_size)
        }
      }
      # ---------------
      # Orientation (D)
      # ---------------
      self$orientation_shared   <- substr(constraint, 3, 3) == "E"
      self$orientation_identity <- substr(constraint, 3, 3) == "I"
      r <- output_dim * (output_dim - 1) / 2
      if (!self$orientation_identity) {
        if (grepl("D", constant_attr)) {
          if (self$orientation_shared) {
            self$D_param <- nn_parameter(0.1 * torch_randn(r))
          } else {
            self$D_param <- nn_parameter(0.1 * torch_randn(n_mixtures, r))
          }
        } else {
          orientation_size <- if (self$orientation_shared) r else (n_mixtures
                                                                   * r)
          self$fc_D <- weight_norm_linear(self$final_hidden_dim,
                                          orientation_size)
        }
      }
      # -----------------------
      # Degrees of freedom (nu)
      # -----------------------
      # Sigmoid: nu = min_nu + (max_nu - min_nu) * sigmoid(raw_nu)
      nu_letter <- substr(constraint, 4, 4)
      self$nu_normal <- (nu_letter == "N")
      self$nu_shared <- (nu_letter == "E" || nu_letter == "N")
      self$nu_fixed <- (nu_letter == "F")  # Flag for fixed nu values
      if (self$nu_fixed) {
        # Validate input
        if (is.null(fixed_nu)) {
          stop("When using constraint with 'F' for fixed nu, you must provide fixed_nu values")
        }
        if (length(fixed_nu) != n_mixtures) {
          stop(paste0("fixed_nu must have length equal to n_mixtures (",
                      n_mixtures, ")"))
        }
        if (is.logical(fixed_nu) && all(is.na(fixed_nu))) {
          fixed_nu <- as.numeric(fixed_nu)
        }
        if (!is.numeric(fixed_nu) || any(is.nan(fixed_nu))) {
          stop("fixed_nu entries must be positive finite values, Inf, or NA.")
        }
        invalid_fixed_nu <- !is.na(fixed_nu) &
          ((is.finite(fixed_nu) & fixed_nu <= 0) |
           (is.infinite(fixed_nu) & fixed_nu < 0))
        if (any(invalid_fixed_nu)) {
          stop("fixed_nu entries must be positive finite values, Inf, or NA.")
        }
        # Store which components have fixed values (not NA)
        fixed_mask <- !is.na(fixed_nu)
        self$register_buffer("fixed_nu_mask", torch_tensor(fixed_mask,
                                                           dtype = torch_bool()))
        # Store fixed values (with NAs replaced by zeros as placeholders)
        fixed_values <- fixed_nu
        fixed_values[is.na(fixed_values)] <- 0
        self$register_buffer("fixed_nu_values",
                             torch_tensor(fixed_values, dtype = torch_float()))
        # Store indices of components that need optimization (NA values)
        self$nu_opt_indices <- which(is.na(fixed_nu))
        # Create parameters for NA components that need optimization
        if (length(self$nu_opt_indices) > 0) {
          nu_size <- length(self$nu_opt_indices)
          # Check if nu should be constant or covariate-dependent
          if (grepl("n", constant_attr)) {
            # Create constant parameters for NA components
            # Initialize raw_nu to 0 (targets middle of range via sigmoid)
            self$nu_param_partial <- nn_parameter(torch_zeros(nu_size))
          } else {
            # Create neural network layer for covariate-dependent nu
            self$fc_nu_partial <- weight_norm_linear(self$final_hidden_dim,
                                                     nu_size)
            # Initialize raw_nu bias to 0 (targets middle of range via sigmoid)
            with_no_grad({
              self$fc_nu_partial$V$normal_(0, 0.02)
              self$fc_nu_partial$g$fill_(0)
              self$fc_nu_partial$bias$fill_(0)
            })
          }
        }
      } else if (!self$nu_normal) {
        nu_size <- if (self$nu_shared) 1 else n_mixtures
        if (grepl("n", constant_attr)) {
          # i.e., nu is "constant" (but distinct across mixture if not shared)
          # Initialize raw_nu to 0 (targets middle of range via sigmoid)
          self$nu_param <- nn_parameter(torch_zeros(nu_size))
        } else {
          # Define a weight-normalized linear layer & custom initialize it
          self$fc_nu <- weight_norm_linear(self$final_hidden_dim, nu_size)
          # Initialize raw_nu bias to 0 (targets middle of range via sigmoid)
          with_no_grad({
            self$fc_nu$V$normal_(0, 0.02)
            self$fc_nu$g$fill_(0)
            self$fc_nu$bias$fill_(0)
          })
        }
      }
      # ----------------
      # Skewness (alpha)
      # ----------------
      skew_letter <- substr(constraint, 5, 5)
      self$skew_none   <- (skew_letter == "N")
      self$skew_shared <- (skew_letter == "E")
      self$skew_vary   <- (!self$skew_none && !self$skew_shared)
      # ---------------  Skewness (alpha)  ---------------
      if (!self$skew_none) {
        alpha_size <- if (self$skew_shared) output_dim else (n_mixtures * output_dim)
        if (grepl("s", constant_attr)) {
          self$alpha_param <- nn_parameter(
            torch_randn(alpha_size, device = self$device, dtype = torch_float()) * 0.05
          )      
        } else {
          self$fc_alpha <- weight_norm_linear(self$final_hidden_dim, alpha_size)      
          with_no_grad({
            self$fc_alpha$V$normal_(0, 0.02)
            self$fc_alpha$bias$zero_()
            self$fc_alpha$g$fill_(0)
          })
        }
      }
    },
    forward = function(x, image_input = NULL) {
      # Process tabular data if a module is provided
      if (!is.null(self$tabular_module)) {
        tabular_features <- self$tabular_module(x)
      } else {
        # Use original input directly if no tabular module
        tabular_features <- x
      }
      # Process image data if available
      if (!is.null(self$image_module) && !is.null(image_input)) {
        image_features <- self$image_module(image_input)
      } else {
        image_features <- NULL
      }
      if (!is.null(self$fusion_module)) {
        if (!is.null(image_features)) {
          h <- self$fusion_module(tabular_features, image_features)
        } else {
          h <- self$fusion_module(tabular_features)
        }
        if (!is.null(self$fusion_dropout)) {
          h <- self$fusion_dropout(h)
        }
      } else {
        if (!is.null(image_features)) {
          # Concatenate features from both branches
          combined_features <- torch_cat(list(tabular_features, image_features),
                                         dim = 2)
        } else {
          # Only tabular features
          combined_features <- tabular_features
        }
        # Continue with existing pipeline using combined features
        h <- self$hidden(combined_features)
      }
      B <- x$size(1)  # batch size
      d <- self$output_dim
      # --------------------
      # Mixture weights (pi)
      # --------------------
      if (grepl("x", self$constant_attr)) {
        pi_logits <- self$pi$unsqueeze(1)$expand(c(B, -1)) # [B, M]
      } else {
        pi_logits <- self$fc_pi(h)                         # [B, M]
      }
      pi_raw <- nnf_softmax(pi_logits, dim = 2)            # [B, M]
      max_weight <- 1.0 - (self$n_mixtures - 1) * self$min_mix_weight
      pi_clamped <- pi_raw$clamp(min = self$min_mix_weight, max = max_weight)
      pi <- pi_clamped / pi_clamped$sum(dim = 2, keepdim = TRUE)
      # ----------
      # Locations (mu)
      # ----------
      if (grepl("m", self$constant_attr)) {
        mu <- self$mu$unsqueeze(1)$expand(c(B, -1, -1)) # [B, M, d]
      } else {
        raw_mu <- self$fc_mu(h)                         # [B, M*d]
        mu <- raw_mu$view(c(B, self$n_mixtures, d))     # [B, M, d]
      }
      # ----------
      # Volume (L)
      # ----------
      if (grepl("L", self$constant_attr)) {
        raw_L <- torch_clamp(self$L_param, min = -20, max = 20)
        L_val <- nnf_softplus(raw_L)$unsqueeze(1)$expand(c(B, -1)) + 1e-6
      } else {
        raw_L <- self$fc_L(h)
        raw_L <- torch_clamp(raw_L, min = -20, max = 20)
        L_val <- nnf_softplus(raw_L) + 1e-6
      }
      if (self$volume_shared) {
        L_val <- L_val$expand(c(-1, self$n_mixtures))  # [B, M]
      }
      L_val <- L_val$unsqueeze(-1)$unsqueeze(-1)       # [B, M, 1, 1]
      # ---------
      # Shape (A)
      # ---------
      if (self$shape_identity) {
        A_diag <- torch_ones(c(B, self$n_mixtures, d), device = x$device)
      } else if (grepl("A", self$constant_attr)) {
        # Clamp logits and use soft-plus(+ε) to prevent under/overflow
        rawA <- self$A_param
        rawA <- torch_clamp(rawA, min = -20, max =  20)
        rawA <- nnf_softplus(rawA) + 1e-6
        if (self$shape_shared) {
          tmp <- rawA$unsqueeze(1)$unsqueeze(1)                # [1,1,d]
          A_diag <- tmp$expand(c(B, self$n_mixtures, -1))      # [B,M,d]
        } else {
          tmp <- rawA$unsqueeze(1)                             # [1,M,d]
          A_diag <- tmp$expand(c(B, -1, -1))                   # [B,M,d]
        }
        # Normalize product
        prodA <- torch_prod(A_diag, dim = -1, keepdim = TRUE)
        A_diag <- A_diag / (prodA^(1 / d))
      } else {
        # Learned shape: same clamp-softplus safeguard
        rawA <- self$fc_A(h)
        rawA <- torch_clamp(rawA, min = -20, max =  20)
        rawA <- nnf_softplus(rawA) + 1e-6
        if (self$shape_shared) {
          A_diag <- rawA$unsqueeze(2)$expand(c(-1, self$n_mixtures, -1))
        } else {
          A_diag <- rawA$view(c(B, self$n_mixtures, d))
        }
        prodA <- torch_prod(A_diag, dim = -1, keepdim = TRUE)
        A_diag <- A_diag / (prodA^(1 / d))
      }
      # Final safety clamp keeps Σ well-conditioned
      A_diag <- torch_clamp(A_diag, min = 1e-3, max = 1e+3)
      # ---------------
      # Orientation (D)
      # ---------------
      if (self$orientation_identity) {
        D_mats <- replicate(
          self$n_mixtures,
          torch_eye(d, device = x$device)$unsqueeze(1)$expand(c(B, -1, -1)),
          simplify = FALSE
        )
      } else {
        if (!is.null(self$fc_D)) {
          rawD <- self$fc_D(h)  # [B, r or B, M*r]
          if (self$orientation_shared) {
            rawD <- rawD$unsqueeze(2)$expand(c(-1, self$n_mixtures, -1))
          } else {
            rawD <- rawD$view(c(B, self$n_mixtures, d * (d - 1) / 2))
          }
          D_mats <- lapply(seq_len(self$n_mixtures), function(j) {
            build_orthogonal_matrix(rawD[, j, ], d)
          })
        } else {
          # D_param
          if (self$orientation_shared) {
            D_exp <- self$D_param$unsqueeze(1)$expand(c(self$n_mixtures, -1))
          } else {
            D_exp <- self$D_param
          }
          D_mats <- lapply(seq_len(self$n_mixtures), function(j) {
            D_j <- build_orthogonal_matrix(D_exp[j, ]$unsqueeze(1), d)
            D_j$expand(c(B, -1, -1))
          })
        }
      }
      D_tensor <- torch_stack(D_mats, dim = 2) # [B, M, d, d]
      # -----------------------
      # Degrees of freedom (nu)
      # -----------------------
      if (self$nu_normal) {
        # The exact Gaussian/skew-normal limit is represented by nu = Inf.
        # Distribution helpers branch before evaluating any Student-t terms.
        nu <- torch_full(
          c(B, self$n_mixtures),
          Inf,
          dtype = x$dtype,
          device = x$device
        )
      } else if (self$nu_fixed) {
        # Get fixed values
        fixed_values <- self$fixed_nu_values
        # Check if any components need optimization
        if (length(self$nu_opt_indices) > 0) {
          # Create a new tensor for nu values
          nu <- torch_zeros(c(B, self$n_mixtures), device = x$device)
          # Fill in the fixed values directly
          fixed_mask_exp <- self$fixed_nu_mask$unsqueeze(1)$expand(c(B, -1))
          nu$masked_scatter_(fixed_mask_exp,
             fixed_values[self$fixed_nu_mask]$unsqueeze(1)$expand(c(B, -1)))
          # Calculate and fill in the optimized values using sigmoid transform
          if (!is.null(self$nu_param_partial)) {
            # Constant optimized values (apply sigmoid transform)
            raw_nu_opt <- self$nu_param_partial # [num_optimized]
            nu_opt <- self$min_nu + (self$max_nu - self$min_nu) *
              torch_sigmoid(raw_nu_opt)
            # Expand nu_opt to batch dimension B
            nu_opt_exp <- nu_opt$unsqueeze(1)$expand(c(B, -1)) # [B, num_optimized]
            # Create mask for optimized indices
            opt_mask <- torch_zeros(c(B, self$n_mixtures), dtype = torch_bool(),
                                    device = x$device)
            opt_mask[, self$nu_opt_indices] <- TRUE
            nu$masked_scatter_(opt_mask, nu_opt_exp) # Fill optimized values
          } else {
            # Covariate-dependent optimized values (apply sigmoid transform)
            raw_nu_opt <- self$fc_nu_partial(h)  # [B, num_optimized]
            nu_opt <- self$min_nu + (self$max_nu - self$min_nu) *
              torch_sigmoid(raw_nu_opt) # [B, num_optimized]
            # Create mask for optimized indices
            opt_mask <- torch_zeros(c(B, self$n_mixtures), dtype = torch_bool(),
                                    device = x$device)
            opt_mask[, self$nu_opt_indices] <- TRUE
            nu$masked_scatter_(opt_mask, nu_opt) # Fill optimized values
          }
        } else {
          # All values are fixed, create a fresh tensor from buffer
          nu <- fixed_values$unsqueeze(1)$expand(c(B, self$n_mixtures))$clone()
        }
      } else if (!is.null(self$nu_param)) {
        # "Constant" learned param (apply sigmoid transform)
        raw_nu <- self$nu_param # [1 or M]
        tmp <- self$min_nu + (self$max_nu - self$min_nu) * torch_sigmoid(raw_nu)
        if (self$nu_shared) {
          # raw_nu is [1], tmp is [1] -> expand to [B, M]
          nu <- tmp$unsqueeze(1)$expand(c(B, self$n_mixtures))
        } else {
          # raw_nu is [M], tmp is [M] -> expand to [B, M]
          nu <- tmp$unsqueeze(1)$expand(c(B, -1))
        }
      } else {
        # Covariate-dependent via fc_nu (apply sigmoid transform)
        raw_nu <- self$fc_nu(h)                    # [B, 1 or B, M]
        tmp <- self$min_nu + (self$max_nu - self$min_nu) * torch_sigmoid(raw_nu)
        if (self$nu_shared) {
          # raw_nu is [B, 1], tmp is [B, 1] -> expand to [B, M]
          nu <- tmp$expand(c(-1, self$n_mixtures))
        } else {
          # raw_nu is [B, M], tmp is [B, M] -> assign directly
          nu <- tmp
        }
      }
      # ----------------
      # Skewness (alpha)
      # ----------------
      if (self$skew_none) {
        alpha <- torch_zeros(c(B, self$n_mixtures, d), device = x$device)
      } else {
        if (!is.null(self$alpha_param)) {
          if (self$skew_shared) {
            alpha <- self$alpha_param$unsqueeze(1)$unsqueeze(1)$expand(
              c(B, self$n_mixtures, d))
          } else {
            alpha_mat <- self$alpha_param$view(c(self$n_mixtures, d))
            alpha <- alpha_mat$unsqueeze(1)$expand(c(B, -1, -1))
          }
        } else {
          raw_alpha <- self$fc_alpha(h)  # [B, M*d or B,d if shared]
          if (self$skew_shared) {
            alpha <- raw_alpha$unsqueeze(2)$expand(c(-1, self$n_mixtures, -1))
          } else {
            alpha <- raw_alpha$view(c(B, self$n_mixtures, d))
          }
        }
      }
      alpha <- self$max_alpha * torch_tanh(alpha)
      # -----------------------------------------
      # Construct scale = L * (D * diag(A) * D^T)
      # -----------------------------------------
      L_val   <- torch_clamp(L_val,   min = self$min_vol_shape, max = 1e2)
      A_diag  <- torch_clamp(A_diag,  min = self$min_vol_shape, max = 1e2)
      # Build Cholesky factor directly
      lambda_half <- torch_sqrt(L_val)
      sqrtA_mats  <- torch_diag_embed(torch_sqrt(A_diag))
      L_direct <- torch_matmul(D_tensor, sqrtA_mats)
      L_direct <- lambda_half * L_direct
      Sigma <- torch_matmul(L_direct, L_direct$transpose(-2, -1))
      eye_mat <- torch_eye(d, device = x$device)$unsqueeze(1)$unsqueeze(1
                           )$expand(c(B, self$n_mixtures, d, d))
      scale_chol <- linalg_cholesky(Sigma + self$jitter * eye_mat)
      # --------------------
      # Return named outputs
      # --------------------
      list(
        pi    = pi,                # [B, M]
        mu    = mu,                # [B, M, d]
        scale_chol = scale_chol,   # [B, M, d, d]
        nu    = nu,                # [B, M]
        alpha = alpha,             # [B, M, d]
        # Volume/Shape/Orientation breakdown
        L     = L_val,   # [B, M, 1, 1]
        A     = A_diag,  # [B, M, d]
        D     = D_tensor # [B, M, d, d]
      )
    }
  )()
}

# --------------------------------------
# PMDN skew t-distribution loss function
# --------------------------------------

loss_mst_pmdn <- function(output, target,
                          lambda_alpha = 0, lambda_nu_inv = 0) {
  # Output must have: pi, mu, scale (Cholesky L), nu, alpha
  # target shape: [B, d]
  pi         <- output$pi         # [B, M]
  mu         <- output$mu         # [B, M, d]
  scale_chol <- output$scale_chol # [B, M, d, d]
  nu         <- output$nu         # [B, M]
  alpha      <- output$alpha      # [B, M, d]
  normal     <- nu == Inf         # exact Gaussian/skew-normal components
  nu_safe    <- torch_where(normal, 3 * torch_ones_like(nu), nu)
  # B <- target$size(1)
  # M <- pi$size(2)
  d <- target$size(2)
  dev <- pi$device # Get device from a parameter
  # Difference: y - mu
  diff <- target$unsqueeze(2) - mu # [B, M, d]
  diff_unsq <- diff$unsqueeze(-1)  # [B, M, d, 1]
  # Solve L v = (y - mu) using Cholesky factor L (scale_chol)
  # Equivalent to v = L^{-1} (y - mu)
  v <- linalg_solve_triangular(scale_chol, diff_unsq,
                               upper = FALSE)$squeeze(-1) # [B, M, d]
  # Mahalanobis distance squared: maha = ||v||^2 = ||L^{-1}(y-mu)||^2
  maha <- v$pow(2)$sum(dim = 3)$clamp(max = 1e6) # [B, M]
  # log|Sigma| = log|L L^T| = 2 * log|L| = 2 * sum(log(diag(L)))
  # Diagonals of Cholesky factor L (scale_chol)
  diag_L <- scale_chol$diagonal(dim1 = -2, dim2 = -1)
  # Clamp diagonal elements before log for stability
  log_det_Sigma <- 2 * diag_L$clamp(min = 1e-12)$log()$sum(dim = 3) # [B, M]
  # Log PDF of each finite-nu multivariate t component. The normalizing
  # constant uses gamma recurrence for even d and float64 intermediates for
  # odd d, avoiding cancellation between large float32 lgamma values.
  half_nu_plus_d <- (nu_safe + d) / 2
  logC_t <- .log_multivariate_t_normalizer(nu_safe, d) -
            0.5 * log_det_Sigma
  logTail <- -half_nu_plus_d * torch_log1p(torch_clamp(maha / nu_safe,
                                           min = -1 + 1e-7, max = 1e7))
  log_pdf_t <- logC_t + logTail

  # Exact nu -> Inf limit of the symmetric kernel.
  log_pdf_normal <- -(d / 2) * log(2 * pi) -
                    0.5 * log_det_Sigma - 0.5 * maha
  log_pdf <- torch_where(normal, log_pdf_normal, log_pdf_t)

  # Skewness factor calculation
  # Clamp large values, [B, M, 1]. The limit is one for normal components.
  cterm <- torch_sqrt((nu_safe + d) / (nu_safe + maha))$clamp(max = 1e6)
  cterm <- torch_where(normal, torch_ones_like(cterm), cterm)$unsqueeze(-1)
  w <- cterm * v # [B, M, d]
  # alpha^T w
  alpha_dot_w <- (alpha * w)$sum(dim = 3) # [B, M]
  # The finite-t skew factor uses T_1; its exact normal limit uses Phi.
  log_skew_cdf <- torch_where(
    normal,
    .log_normal_cdf(alpha_dot_w),
    log_pt(alpha_dot_w, nu_safe + d)
  )
  log_skew_factor <- torch_log(torch_tensor(2.0, device = dev)) +
                     log_skew_cdf
  log_component <- log_pdf + log_skew_factor # [B, M]
  # Mixture weighting and log-sum-exp for total log-likelihood
  # log P(y|x) = log sum_k [ pi_k * SkewT(y | mu_k, Sigma_k, alpha_k, nu_k) ]
  #            = logsumexp_k [ log(pi_k) + log(SkewT(...)) ]
  weighted_log_probs <- torch_log(pi$clamp(min = 1e-12)) +
                        log_component # [B, M]
  # Negative log-likelihood (average over batch)
  loss <- -torch_logsumexp(weighted_log_probs, dim = 2)$mean()
  # L2 penalty on final alpha values
  loss <- loss + lambda_alpha * alpha$pow(2)$mean()
  # (1/nu)^2 penalty on degrees of freedom
  nu_inv_sq <- torch_where(
    normal,
    torch_zeros_like(nu_safe),
    nu_safe$pow(-2)
  )
  loss <- loss + lambda_nu_inv * nu_inv_sq$mean()
  loss
}

# -----------------------------------------------
# Skew t-distribution random sampling (on device)
# *_df version converts to R data frame
# -----------------------------------------------

sample_mst_pmdn <- function(mdn_output, num_samples = 1, device = "cpu") {
  num_samples <- validate_num_samples(num_samples)
  # gather parameters
  pi     <- mdn_output$pi          $to(device = device)
  mu     <- mdn_output$mu          $to(device = device)
  L_all  <- mdn_output$scale_chol  $to(device = device)
  nu_all <- mdn_output$nu          $to(device = device)
  alpha_all <- mdn_output$alpha    $to(device = device)
  B <- pi$size(1)
  # M <- pi$size(2)
  d <- mu$size(3)
  # Use the public torch wrapper, which converts libtorch's zero-based result
  # to the one-based indices expected by R torch indexing operations.
  idx      <- torch_multinomial(pi, num_samples, replacement = TRUE)
  idx_d    <- idx$unsqueeze(-1)$expand(c(B, num_samples, d))
  idx_dd   <- idx$unsqueeze(-1)$unsqueeze(-1)$expand(c(B, num_samples, d, d))
  # gather parameters for the selected components
  mu_s    <- mu       $gather(2, idx_d)
  L_s     <- L_all    $gather(2, idx_dd)
  nu_s    <- nu_all   $gather(2, idx)
  alpha_s <- alpha_all$gather(2, idx_d)
  # Gamma scaling for Student-t tails. Exact normal components use W = 1;
  # a finite placeholder prevents Inf from entering the unused Gamma branch.
  normal_s <- nu_s == Inf
  if (normal_s$all()$item()) {
    W <- torch_ones_like(nu_s)$unsqueeze(-1)
  } else {
    nu_sample <- torch_where(normal_s, 2 * torch_ones_like(nu_s), nu_s)
    chi2 <- sample_gamma(nu_sample / 2, scale = 2, device = device)
    W_t <- torch_sqrt(nu_sample / chi2$clamp(min = 1e-12))
    W <- torch_where(normal_s, torch_ones_like(W_t), W_t)$unsqueeze(-1)
  }
  # skew direction (identity-covariance, Sigma = I convention)
  alpha_norm_sq <- alpha_s$pow(2)$sum(dim = -1, keepdim = TRUE)
  delta <- alpha_s / torch_sqrt(1 + alpha_norm_sq)
  delta_norm_sq <- delta$pow(2)$sum(dim = -1, keepdim = TRUE)
  # standard normals
  z0 <- torch_randn(c(B, num_samples, 1), device = device)
  z1 <- torch_randn(c(B, num_samples, d), device = device)
  # Skew-normal core.  The residual requires the symmetric matrix square root
  # of I - delta delta^T.  Apply its rank-one form without materializing a
  # [B, S, d, d] tensor:
  # (I - delta delta^T)^(1/2) z =
  # z - delta (delta^T z) / (1 + sqrt(1 - ||delta||^2)).
  sqrt_one_minus_delta_sq <- torch_sqrt(
    (1 - delta_norm_sq)$clamp(min = 1e-12)
  )
  delta_dot_z1 <- (delta * z1)$sum(dim = -1, keepdim = TRUE)
  residual <- z1 - delta * delta_dot_z1 /
    (1 + sqrt_one_minus_delta_sq)
  X <- delta * torch_abs(z0) + residual
  # affine map to response space  Y
  Y <- mu_s + W * (torch_matmul(L_s, X$unsqueeze(-1))$squeeze(-1))
  # return both samples and component IDs
  list(
    samples    = Y$permute(c(2, 1, 3)),
    components = idx$permute(c(2, 1))
  )
}

sample_mst_pmdn_df <- function(mdn_output, num_samples = 1, device = "cpu") {
  num_samples <- validate_num_samples(num_samples)
  sampled <- sample_mst_pmdn(
    mdn_output,
    num_samples = num_samples,
    device = device
  )
  # Restore [B, S, ...] layout for data-frame construction.
  Y <- sampled$samples$permute(c(2, 1, 3))
  idx <- sampled$components$permute(c(2, 1))
  B <- Y$size(1)
  d <- Y$size(3)
  # reshape to long data-frame
  S   <- num_samples
  mat  <- as.matrix(Y$reshape(c(B * S, d))$cpu())
  comp <- as.integer(idx$reshape(c(B * S))$cpu())
  data.frame(mat,
             row  = rep(seq_len(B), each = S),
             draw = rep(seq_len(S), times = B),
             comp = factor(comp))
}

# -------------------------------------------------------
# Monte Carlo approximation to marginal CDFs via sampling
# -------------------------------------------------------

cdf_marginal_mst_pmdn <- function(mdn_output,
                                 y,
                                 var_index = NULL,
                                 num_samples = 1000,
                                 device = "cpu",
                                 seed = NULL,
                                 draws = NULL) {
  if (!is.list(mdn_output)) {
    stop("mdn_output must be a list returned by predict_mst_pmdn.")
  }
  required_fields <- c("pi", "mu", "scale_chol", "nu", "alpha")
  if (!all(required_fields %in% names(mdn_output))) {
    stop("mdn_output is missing required fields from predict_mst_pmdn.")
  }
  num_samples <- validate_num_samples(num_samples)
  d <- mdn_output$mu$size(3)
  var_index_provided <- !is.null(var_index)
  if (!var_index_provided) {
    var_index <- seq_len(d)
  }
  var_index <- as.integer(var_index)
  if (length(var_index) == 0 || any(is.na(var_index))) {
    stop("var_index must contain valid dimension indices.")
  }
  if (any(var_index < 1) || any(var_index > d)) {
    stop("var_index is out of bounds for output dimensions.")
  }
  B <- mdn_output$pi$size(1)
  if (!inherits(y, "torch_tensor")) {
    x_tensor <- torch_tensor(y, device = device, dtype = torch_float())
  } else {
    x_tensor <- y$to(device = device, dtype = torch_float())
  }
  x_dim <- x_tensor$size()
  if (length(x_dim) == 0) {
    x_tensor <- x_tensor$reshape(c(1, 1))$expand(c(B, length(var_index)))
  } else if (length(x_dim) == 1) {
    if (x_dim[1] == d) {
      x_tensor <- x_tensor$reshape(c(1, d))$expand(c(B, d))
    } else if (x_dim[1] == length(var_index)) {
      x_tensor <- x_tensor$reshape(c(1, length(var_index)))$
        expand(c(B, length(var_index)))
    } else if (x_dim[1] == 1) {
      x_tensor <- x_tensor$reshape(c(1, 1))$expand(c(B, length(var_index)))
    } else {
      stop("y must be a scalar, length d, or length(var_index).")
    }
  } else if (length(x_dim) == 2) {
    if (x_dim[1] != B) {
      stop("y must have B rows to match batch size.")
    }
    if (x_dim[2] != d && x_dim[2] != length(var_index)) {
      stop("y must have d columns or length(var_index) columns.")
    }
  } else {
    stop("y must be a vector or matrix.")
  }
  if (x_tensor$size(2) == d) {
    x_tensor <- x_tensor[, var_index]
  }
  if (!is.null(draws)) {
    if (is.list(draws) && !is.null(draws$samples)) {
      draws <- draws$samples
    }
    if (!inherits(draws, "torch_tensor")) {
      if (!is.array(draws)) {
        stop("draws must be a torch tensor, array, or list with a samples entry.")
      }
      draws <- torch_tensor(draws, device = device, dtype = torch_float())
    } else {
      draws <- draws$to(device = device, dtype = torch_float())
    }
    if (length(draws$size()) != 3) {
      stop("draws must have three dimensions (num_samples x batch x output_dim).")
    }
    if (draws$size(2) != B) {
      stop("draws must have the same batch size as mdn_output.")
    }
    if (draws$size(3) != d) {
      stop("draws must have the same output dimension as mdn_output.")
    }
    samples <- draws[, , var_index]
  } else {
    if (!is.null(seed)) {
      set.seed(seed)
      if (exists("torch_manual_seed", mode = "function")) {
        torch_manual_seed(seed)
      }
    }
    draws <- sample_mst_pmdn(mdn_output, num_samples, device = device)
    samples <- draws$samples[, , var_index]
  }
  x_broadcast <- x_tensor$unsqueeze(1)
  counts <- (samples <= x_broadcast)$to(dtype = torch_float())$sum(dim = 1)
  n_draws <- samples$size(1)
  cdf <- (counts / n_draws)$clamp(min = 0, max = 1)
  cdf
}

# -------------------------------------------------
# Monte Carlo marginal quantiles from MST-PMDN
# -------------------------------------------------

quantile_marginal_mst_pmdn <- function(mdn_output,
                                       probs,
                                       var_index = NULL,
                                       num_samples = 1000,
                                       device = "cpu",
                                       seed = NULL,
                                       draws = NULL) {
  probs_is_torch <- inherits(probs, "torch_tensor")
  probs_matrix <- probs
  if (probs_is_torch) {
    probs_matrix <- as.array(probs$to(device = "cpu"))
  }
  if (!is.matrix(probs_matrix)) {
    stop("probs must be a numeric matrix or torch tensor with shape B x V.")
  }
  if (!is.numeric(probs_matrix) || any(probs_matrix < 0 | probs_matrix > 1, na.rm = TRUE)) {
    stop("probs must be numeric with all values in [0, 1].")
  }
  var_index_provided <- !is.null(var_index)
  if (!var_index_provided) {
    var_index <- seq_len(mdn_output$mu$size(3))
  }
  if (!is.numeric(var_index) || any(var_index < 1)) {
    stop("var_index must be a numeric vector of positive indices.")
  }
  if (!is.null(draws)) {
    if (is.list(draws) && !is.null(draws$samples)) {
      draws <- draws$samples
    }
    if (!inherits(draws, "torch_tensor")) {
      if (!is.array(draws)) {
        stop("draws must be a torch tensor, array, or list with a samples entry.")
      }
      arr <- draws
    } else {
      draws <- draws$to(device = "cpu")
      arr <- as.array(draws)
    }
    if (length(dim(arr)) != 3) {
      stop("draws must have three dimensions (num_samples x batch x output_dim).")
    }
  } else {
    if (!is.null(seed)) {
      set.seed(seed)
      if (exists("torch_manual_seed", mode = "function")) {
        torch_manual_seed(seed)
      }
    }
    draws <- sample_mst_pmdn(mdn_output,
                             num_samples = num_samples,
                             device = device)$samples
    draws <- draws$to(device = "cpu")
    arr <- as.array(draws)
  }
  B <- dim(arr)[2]
  V <- length(var_index)
  if (nrow(probs_matrix) != B) {
    stop("probs must have the same number of rows as the batch size.")
  }
  if (dim(arr)[2] != mdn_output$pi$size(1)) {
    stop("draws must have the same batch size as mdn_output.")
  }
  if (dim(arr)[3] != mdn_output$mu$size(3)) {
    stop("draws must have the same output dimension as mdn_output.")
  }
  if (any(var_index > dim(arr)[3])) {
    stop("var_index contains indices larger than the output dimension.")
  }
  if (var_index_provided && ncol(probs_matrix) != length(var_index)) {
    stop("When provided, var_index must have the same length as ncol(probs).")
  }
  if (!var_index_provided && ncol(probs_matrix) != dim(arr)[3]) {
    stop("When var_index is NULL, ncol(probs) must match the output dimension.")
  }
  arr <- arr[, , var_index, drop = FALSE]
  compute_quantile <- function(samples, p) {
    samples <- samples[!is.na(samples)]
    n <- length(samples)
    if (n == 0) {
      return(NA_real_)
    }
    sorted_samples <- sort(samples)
    position <- (n - 1) * p + 1
    lower <- floor(position)
    upper <- ceiling(position)
    if (lower == upper) {
      sorted_samples[lower]
    } else {
      weight <- position - lower
      (1 - weight) * sorted_samples[lower] + weight * sorted_samples[upper]
    }
  }
  out <- matrix(NA_real_, nrow = B, ncol = V,
                dimnames = list(batch = seq_len(B), var = var_index))
  for (b in seq_len(B)) {
    for (v in seq_len(V)) {
      out[b, v] <- compute_quantile(arr[, b, v], probs_matrix[b, v])
    }
  }
  out
}

# -------------------------------------------------
# PMDN training function with optional image inputs
# -------------------------------------------------

train_mst_pmdn <- function(inputs,
                           outputs,
                           hidden_dim,
                           n_mixtures,
                           constraint = "VVVNN",
                           constant_attr = "",
                           fixed_nu = NULL,
                           range_nu = c(3., 50.),
                           max_alpha = 2.5,
                           min_vol_shape = 1e-2,
                           min_mix_weight = 1e-4,
                           jitter = 1e-6,
                           activation = nn_tanh,
                           lambda_alpha = 0,
                           lambda_nu_inv = 0,
                           epochs = 500,
                           lr = 0.001,
                           batch_size = 16,
                           max_norm = 1.,
                           drop_hidden = 0.,
                           wd_image = 0.,
                           wd_tabular = 0.,
                           checkpoint_interval = 10,
                           checkpoint_path = "checkpoint.pt",
                           resume_from_checkpoint = FALSE,
                           model = NULL,
                           early_stopping_patience = 50,
                           validation_split = 0.2,
                           custom_split = NULL,
                           scheduler_step = 50,
                           scheduler_gamma = 0.5,
                           image_inputs = NULL,
                           image_module = NULL,
                           tabular_module = NULL,
                           fusion_module = NULL,
                           min_last_batch_frac = 0.5,
                           device = "cpu"
) {
  is_positive_integer <- function(x) {
    is.numeric(x) && length(x) == 1 && is.finite(x) &&
      x >= 1 && x == floor(x)
  }
  if (!is_positive_integer(batch_size))
    stop("batch_size must be a single positive integer")
  if (!is_positive_integer(epochs))
    stop("epochs must be a single positive integer")
  if (!is_positive_integer(checkpoint_interval))
    stop("checkpoint_interval must be a single positive integer")
  if (!is.null(scheduler_step) && !is_positive_integer(scheduler_step))
    stop("scheduler_step must be NULL or a single positive integer")
  batch_size <- as.integer(batch_size)
  epochs <- as.integer(epochs)
  checkpoint_interval <- as.integer(checkpoint_interval)
  if (!is.null(scheduler_step)) scheduler_step <- as.integer(scheduler_step)
  if (!is.numeric(min_last_batch_frac) || length(min_last_batch_frac) != 1 ||
      !is.finite(min_last_batch_frac) || min_last_batch_frac < 0 ||
      min_last_batch_frac > 1)
    stop("min_last_batch_frac must be a single number between 0 and 1")
  if (!is.numeric(validation_split) || length(validation_split) != 1 ||
      !is.finite(validation_split) || validation_split < 0 ||
      validation_split >= 1)
    stop("validation_split must be a single number in [0, 1)")
  if (isTRUE(resume_from_checkpoint) && !is.null(model))
    stop("model and resume_from_checkpoint cannot be used together")
  if (isTRUE(resume_from_checkpoint) && !file.exists(checkpoint_path))
    stop("resume_from_checkpoint is TRUE but checkpoint_path does not exist")

  # Data preparation
  if (!inherits(inputs, "torch_tensor"))
    inputs <- torch_tensor(inputs, device = device, dtype = torch_float())
  else
    inputs <- inputs$to(device = device)
  if (!inherits(outputs, "torch_tensor"))
    outputs <- torch_tensor(outputs, device = device, dtype = torch_float())
  else
    outputs <- outputs$to(device = device)
  if (!is.null(image_inputs)) {
    if (!inherits(image_inputs, "torch_tensor"))
      image_inputs <- torch_tensor(image_inputs, device = device,
                                   dtype = torch_float())
    else
      image_inputs <- image_inputs$to(device = device)
  }
  n_total <- inputs$size(1)
  input_dim <- inputs$size(2)
  output_dim <- outputs$size(2)
  if (outputs$size(1) != n_total)
    stop("inputs and outputs must have the same number of rows")
  if (!is.null(image_inputs) && image_inputs$size(1) != n_total)
    stop("image_inputs must have the same number of rows as inputs")

  resuming <- isTRUE(resume_from_checkpoint)
  checkpoint <- if (resuming) torch_load(checkpoint_path) else NULL
  if (resuming && !is.null(checkpoint$data_signature)) {
    signature <- checkpoint$data_signature
    if (!identical(as.integer(signature$n_total), as.integer(n_total)) ||
        !identical(as.integer(signature$input_dim), as.integer(input_dim)) ||
        !identical(as.integer(signature$output_dim), as.integer(output_dim)) ||
        !identical(as.integer(signature$n_mixtures), as.integer(n_mixtures)) ||
        !identical(signature$constraint, constraint) ||
        !identical(signature$constant_attr, constant_attr)) {
      stop("Checkpoint architecture or data dimensions do not match this run")
    }
  }

  # Restore the original split before making data subsets. Older checkpoints
  # lack these fields and retain the legacy split behavior with a warning.
  if (resuming && !is.null(checkpoint$train_indices) &&
      !is.null(checkpoint$val_indices)) {
    train_indices <- checkpoint$train_indices
    val_indices <- checkpoint$val_indices
    if (!is.null(custom_split)) {
      warning("custom_split is ignored when resuming; using the split saved in the checkpoint.")
    }
  } else {
    if (resuming) {
      warning("Legacy checkpoint has no saved split; exact resumption is not possible.")
    }
    if (!is.null(custom_split)) {
      if (is.list(custom_split) &&
          all(c("train", "validation") %in% names(custom_split))) {
        train_indices <- custom_split$train
        val_indices <- custom_split$validation
      } else if (is.list(custom_split) && length(custom_split) == 2) {
        train_indices <- custom_split[[1]]
        val_indices <- custom_split[[2]]
      } else if (is.numeric(custom_split)) {
        val_indices <- custom_split
        train_indices <- setdiff(seq_len(n_total), val_indices)
      } else if (is.logical(custom_split) && length(custom_split) == n_total) {
        train_indices <- which(custom_split)
        val_indices <- which(!custom_split)
      } else {
        stop("Invalid custom_split format.")
      }
    } else if (validation_split > 0) {
      n_validation <- floor(n_total * validation_split)
      if (n_validation < 1)
        stop("validation_split produces an empty validation set")
      val_indices <- sample.int(n_total, size = n_validation)
      train_indices <- setdiff(seq_len(n_total), val_indices)
    } else {
      train_indices <- seq_len(n_total)
      val_indices <- integer(0)
    }
  }
  train_indices <- as.integer(train_indices)
  val_indices <- as.integer(val_indices)
  if (length(train_indices) == 0 || anyNA(train_indices) ||
      any(train_indices < 1 | train_indices > n_total) ||
      anyDuplicated(train_indices))
    stop("Training indices must be unique, in bounds, and non-empty")
  if (anyNA(val_indices) || any(val_indices < 1 | val_indices > n_total) ||
      anyDuplicated(val_indices))
    stop("Validation indices must be unique and in bounds")
  if (length(intersect(train_indices, val_indices)) > 0)
    stop("Training and validation indices must be disjoint")

  train_inputs <- inputs[train_indices, ]
  train_outputs <- outputs[train_indices, ]
  train_image_inputs <- if (!is.null(image_inputs)) {
    image_inputs[train_indices, ]
  } else {
    NULL
  }
  if (length(val_indices) > 0) {
    val_inputs <- inputs[val_indices, ]
    val_outputs <- outputs[val_indices, ]
    val_image_inputs <- if (!is.null(image_inputs)) {
      image_inputs[val_indices, ]
    } else {
      NULL
    }
  } else {
    val_inputs <- NULL
    val_outputs <- NULL
    val_image_inputs <- NULL
  }

  extend_history <- function(history, length_out) {
    if (is.null(history)) history <- numeric(0)
    if (length(history) < length_out) {
      history <- c(history, rep(NA_real_, length_out - length(history)))
    }
    history
  }
  history_length <- max(epochs, if (resuming) checkpoint$epoch else 0)

  # Model initialization logic
  if (!is.null(model)) {
    model <- model$to(device = device)
    start_epoch <- 1L
    train_loss_history <- rep(NA_real_, history_length)
    val_loss_history <- rep(NA_real_, history_length)
    best_val_loss <- Inf
    best_val_epoch <- NA_integer_
    no_improve_epochs <- 0L
    best_train_loss <- Inf
    best_train_epoch <- NA_integer_
  } else if (resuming) {
    model <- define_mst_pmdn(
      input_dim, output_dim, hidden_dim, n_mixtures,
      constraint, constant_attr,
      activation = activation,
      drop_hidden = drop_hidden,
      image_module = image_module,
      tabular_module = tabular_module,
      fusion_module = fusion_module,
      fixed_nu = fixed_nu,
      range_nu = range_nu,
      max_alpha = max_alpha,
      min_vol_shape = min_vol_shape,
      min_mix_weight = min_mix_weight,
      jitter = jitter
    )
    model$load_state_dict(checkpoint$model_state_dict)
    model <- model$to(device = device)
    train_loss_history <- extend_history(
      checkpoint$train_loss_history, history_length
    )
    val_loss_history <- extend_history(
      checkpoint$val_loss_history, history_length
    )
    start_epoch <- as.integer(checkpoint$epoch) + 1L
    best_val_loss <- if (!is.null(checkpoint$best_val_loss)) {
      checkpoint$best_val_loss
    } else {
      Inf
    }
    best_val_epoch <- if (!is.null(checkpoint$best_val_epoch)) {
      checkpoint$best_val_epoch
    } else {
      NA_integer_
    }
    no_improve_epochs <- if (!is.null(checkpoint$no_improve_epochs)) {
      checkpoint$no_improve_epochs
    } else {
      0L
    }
    best_train_loss <- if (!is.null(checkpoint$best_train_loss)) {
      checkpoint$best_train_loss
    } else {
      Inf
    }
    best_train_epoch <- if (!is.null(checkpoint$best_train_epoch)) {
      checkpoint$best_train_epoch
    } else {
      NA_integer_
    }
    cat(sprintf("Resumed from checkpoint at epoch %d.\n", checkpoint$epoch))
  } else {
    model <- define_mst_pmdn(
      input_dim, output_dim, hidden_dim, n_mixtures,
      constraint, constant_attr,
      activation = activation,
      drop_hidden = drop_hidden,
      image_module = image_module,
      tabular_module = tabular_module,
      fusion_module = fusion_module,
      fixed_nu = fixed_nu,
      range_nu = range_nu,
      max_alpha = max_alpha,
      min_vol_shape = min_vol_shape,
      min_mix_weight = min_mix_weight,
      jitter = jitter
    )
    model$apply(init_weight_norm)
    init_distribution_heads(model)
    model <- model$to(device = device)
    init_mu_kmeans(
      model,
      outputs_train = train_outputs,
      n_mixtures = n_mixtures,
      constant_attr = constant_attr,
      device = device
    )
    start_epoch <- 1L
    train_loss_history <- rep(NA_real_, history_length)
    val_loss_history <- rep(NA_real_, history_length)
    best_val_loss <- Inf
    best_val_epoch <- NA_integer_
    no_improve_epochs <- 0L
    best_train_loss <- Inf
    best_train_epoch <- NA_integer_
  }

  # Adam optimizer
  img_params <- if (!is.null(model$image_module)) {
    model$image_module$parameters
  } else list()
  tab_params <- if (!is.null(model$tabular_module)) {
    model$tabular_module$parameters
  } else list()
  fusion_params <- if (!is.null(model$fusion_module)) {
    model$fusion_module$parameters
  } else list()
  hidden_params <- if (!is.null(model$hidden)) model$hidden$parameters else list()
  all_params <- model$parameters
  feat_params <- c(img_params, tab_params, fusion_params, hidden_params)
  head_params <- setdiff(all_params, feat_params)
  optimizer <- optim_adam(
    params = list(
      list(params = img_params, weight_decay = wd_image),
      list(params = tab_params, weight_decay = wd_tabular),
      list(params = fusion_params),
      list(params = hidden_params),
      list(params = head_params)
    ),
    lr = lr
  )
  if (resuming) {
    if (is.null(checkpoint$optimizer_state_dict)) {
      warning("Checkpoint has no optimizer state; optimizer starts fresh.")
    } else {
      optimizer$load_state_dict(checkpoint$optimizer_state_dict)
    }
  }

  best_checkpoint_path <- derive_checkpoint_path(checkpoint_path, "_best")
  best_train_checkpoint_path <- derive_checkpoint_path(
    checkpoint_path, "_trainbest"
  )

  make_checkpoint <- function(epoch) {
    list(
      epoch = as.integer(epoch),
      model_state_dict = model$state_dict(),
      optimizer_state_dict = optimizer$state_dict(),
      best_val_loss = best_val_loss,
      best_val_epoch = best_val_epoch,
      no_improve_epochs = no_improve_epochs,
      best_train_loss = best_train_loss,
      best_train_epoch = best_train_epoch,
      train_loss_history = train_loss_history,
      val_loss_history = val_loss_history,
      train_indices = train_indices,
      val_indices = val_indices,
      data_signature = list(
        n_total = n_total,
        input_dim = input_dim,
        output_dim = output_dim,
        n_mixtures = n_mixtures,
        constraint = constraint,
        constant_attr = constant_attr
      ),
      rng_state = capture_training_rng_state()
    )
  }

  # Dataloaders
  dataset_fn <- function(inp, img_inp, outp) {
    if (is.null(img_inp)) {
      dataset(
        initialize = function(x, y) { self$x <- x; self$y <- y },
        .getitem = function(idx) list(self$x[idx, ], self$y[idx, ]),
        .length = function() self$x$size(1)
      )(inp, outp)
    } else {
      dataset(
        initialize = function(x, im, y) {
          self$x <- x; self$im <- im; self$y <- y
        },
        .getitem = function(idx) {
          list(self$x[idx, ], self$im[idx, ], self$y[idx, ])
        },
        .length = function() self$x$size(1)
      )(inp, img_inp, outp)
    }
  }
  should_drop_last_batch <- function(n_samples, batch_size, min_frac) {
    remainder <- n_samples %% batch_size
    min_last_batch_size <- ceiling(batch_size * min_frac)
    n_samples > batch_size && remainder > 0 &&
      remainder < min_last_batch_size
  }
  train_dataset <- dataset_fn(train_inputs, train_image_inputs, train_outputs)
  drop_last_train <- should_drop_last_batch(
    length(train_indices), batch_size, min_last_batch_frac
  )
  train_loader <- dataloader(
    train_dataset,
    batch_size = batch_size,
    shuffle = TRUE,
    drop_last = drop_last_train
  )
  if (!is.null(val_inputs)) {
    val_dataset <- dataset_fn(val_inputs, val_image_inputs, val_outputs)
    val_loader <- dataloader(
      val_dataset,
      batch_size = batch_size,
      shuffle = FALSE,
      drop_last = FALSE
    )
  }
  if (resuming && !restore_training_rng_state(checkpoint$rng_state)) {
    warning("Legacy checkpoint has no RNG state; exact resumption is not possible.")
  }

  # Training loop. Loss histories are observation-weighted so their values do
  # not depend on the size of the final batch.
  completed_epoch <- if (resuming) as.integer(checkpoint$epoch) else 0L
  epoch_sequence <- if (start_epoch <= epochs) {
    seq.int(start_epoch, epochs)
  } else {
    integer(0)
  }
  for (epoch in epoch_sequence) {
    model$train()
    total_loss <- 0
    train_cases <- 0L
    coro::loop(for (batch in train_loader) {
      optimizer$zero_grad()
      if (length(batch) == 3) {
        inputs_batch <- batch[[1]]
        image_inputs_batch <- batch[[2]]
        outputs_batch <- batch[[3]]
        pred <- model(inputs_batch, image_inputs_batch)
      } else {
        inputs_batch <- batch[[1]]
        outputs_batch <- batch[[2]]
        pred <- model(inputs_batch)
      }
      loss <- loss_mst_pmdn(
        pred,
        outputs_batch,
        lambda_alpha = lambda_alpha,
        lambda_nu_inv = lambda_nu_inv
      )
      loss$backward()
      if (!is.null(max_norm))
        nn_utils_clip_grad_norm_(model$parameters, max_norm)
      optimizer$step()
      batch_cases <- outputs_batch$size(1)
      total_loss <- total_loss + loss$item() * batch_cases
      train_cases <- train_cases + batch_cases
    })
    if (train_cases == 0)
      stop("No training cases were evaluated")
    avg_train_loss <- total_loss / train_cases
    train_loss_history[epoch] <- avg_train_loss
    completed_epoch <- epoch

    if (!is.null(val_inputs)) {
      model$eval()
      total_val_loss <- 0
      val_cases <- 0L
      with_no_grad({
        coro::loop(for (batch in val_loader) {
          if (length(batch) == 3) {
            inputs_batch <- batch[[1]]
            image_inputs_batch <- batch[[2]]
            outputs_batch <- batch[[3]]
            pred <- model(inputs_batch, image_inputs_batch)
          } else {
            inputs_batch <- batch[[1]]
            outputs_batch <- batch[[2]]
            pred <- model(inputs_batch)
          }
          loss <- loss_mst_pmdn(
            pred,
            outputs_batch,
            lambda_alpha = lambda_alpha,
            lambda_nu_inv = lambda_nu_inv
          )
          batch_cases <- outputs_batch$size(1)
          total_val_loss <- total_val_loss + loss$item() * batch_cases
          val_cases <- val_cases + batch_cases
        })
      })
      if (val_cases != length(val_indices))
        stop("Validation loader did not evaluate every validation case")
      avg_val_loss <- total_val_loss / val_cases
      val_loss_history[epoch] <- avg_val_loss
      cat(sprintf(
        "Epoch %d - Train Loss: %.4f - Val Loss: %.4f\n",
        epoch, avg_train_loss, avg_val_loss
      ))
      if (avg_val_loss < best_val_loss) {
        best_val_loss <- avg_val_loss
        best_val_epoch <- epoch
        no_improve_epochs <- 0L
        torch_save(make_checkpoint(epoch), best_checkpoint_path)
        cat(sprintf(
          "Best checkpoint saved at epoch %d (best_val_loss=%.4f).\n",
          epoch, best_val_loss
        ))
      } else {
        no_improve_epochs <- no_improve_epochs + 1L
      }
    } else {
      cat(sprintf("Epoch %d - Train Loss: %.4f\n", epoch, avg_train_loss))
      if (avg_train_loss < best_train_loss) {
        best_train_loss <- avg_train_loss
        best_train_epoch <- epoch
        torch_save(make_checkpoint(epoch), best_train_checkpoint_path)
        cat(sprintf(
          "Best training checkpoint saved at epoch %d (loss=%.4f).\n",
          epoch, best_train_loss
        ))
      }
    }

    # Apply the scheduled decay before saving the resumable state so the
    # restored optimizer has the learning rate for the following epoch.
    if (!is.null(scheduler_step) && epoch %% scheduler_step == 0) {
      for (group in optimizer$param_groups)
        group$lr <- group$lr * scheduler_gamma
      cat(sprintf("Learning rate updated at epoch %d.\n", epoch))
    }
    if (epoch %% checkpoint_interval == 0) {
      torch_save(make_checkpoint(epoch), checkpoint_path)
      cat(sprintf("Latest checkpoint saved at epoch %d.\n", epoch))
    }
    if (!is.null(val_inputs) &&
        no_improve_epochs >= early_stopping_patience) {
      torch_save(make_checkpoint(epoch), checkpoint_path)
      cat(sprintf(
        paste0("Early stopping triggered at epoch %d ",
               "(no improvement for %d epochs).\n"),
        epoch, early_stopping_patience
      ))
      break
    }
  }

  final_epoch <- completed_epoch
  if (final_epoch < 1)
    stop("No training epoch is available")
  # Always leave a current, resumable checkpoint before loading the best model.
  torch_save(make_checkpoint(final_epoch), checkpoint_path)

  if (!is.null(val_inputs) && file.exists(best_checkpoint_path)) {
    best_checkpoint <- torch_load(best_checkpoint_path)
    model$load_state_dict(best_checkpoint$model_state_dict)
    best_val_loss <- best_checkpoint$best_val_loss
    best_val_epoch <- best_checkpoint$best_val_epoch
    cat("Best model loaded from validation-based checkpoint.\n")
  } else if (is.null(val_inputs) &&
             file.exists(best_train_checkpoint_path)) {
    best_checkpoint <- torch_load(best_train_checkpoint_path)
    model$load_state_dict(best_checkpoint$model_state_dict)
    best_train_loss <- best_checkpoint$best_train_loss
    best_train_epoch <- best_checkpoint$best_train_epoch
    cat("Best model loaded from training-based checkpoint.\n")
  } else {
    warning("No best-model checkpoint was available; returning the latest model.")
  }

  list(
    model = model,
    train_loss_history = train_loss_history[seq_len(final_epoch)],
    val_loss_history = if (!is.null(val_inputs)) {
      val_loss_history[seq_len(final_epoch)]
    } else {
      NULL
    },
    best_val_epoch = best_val_epoch,
    best_val_loss = if (!is.null(val_inputs)) best_val_loss else NULL,
    best_train_epoch = if (is.null(val_inputs)) best_train_epoch else NULL,
    best_train_loss = if (is.null(val_inputs)) best_train_loss else NULL,
    final_epoch = final_epoch,
    train_indices = train_indices,
    val_indices = val_indices,
    checkpoint_path = checkpoint_path,
    best_checkpoint_path = if (!is.null(val_inputs)) {
      best_checkpoint_path
    } else {
      best_train_checkpoint_path
    }
  )
}

# --------------------------------------------------
# PMDN inference function with optional image inputs
# --------------------------------------------------

predict_mst_pmdn <- function(model, new_inputs, image_inputs = NULL,
                             device = "cpu") {
  model$eval()
  if (!inherits(new_inputs, "torch_tensor")) {
    new_inputs <- torch_tensor(new_inputs, device = device,
                               dtype = torch_float())
  } else {
    new_inputs <- new_inputs$to(device = device)
  }
  if (!is.null(image_inputs)) {
    if (!inherits(image_inputs, "torch_tensor")) {
      image_inputs <- torch_tensor(image_inputs, device = device,
                                   dtype = torch_float())
    } else {
      image_inputs <- image_inputs$to(device = device)
    }
    with_no_grad({
      model(new_inputs, image_inputs)
    })
  } else {
    with_no_grad({
      model(new_inputs)
    })
  }
}

scov_mst_pmdn <- function(pred, type = c("scale", "scale_chol", "cov"),
                          as_array = FALSE) {
  # Compute the scale matrix, its Cholesky factor, or the actual component
  # covariance matrix from a list returned by predict_mst_pmdn.
  # as_array: TRUE to return an R array instead of a torch_tensor
  type <- match.arg(type)
  required_fields <- c("scale_chol", "nu", "alpha")
  if (!all(required_fields %in% names(pred))) {
    stop("pred must contain scale_chol, nu, and alpha.")
  }
  scale_chol <- pred$scale_chol
  device <- scale_chol$device
  nu <- pred$nu$to(device = device)
  alpha <- pred$alpha$to(device = device)
  d <- scale_chol$size(3)
  if (type == "scale_chol") {
    out <- scale_chol
  } else {
    scale <- torch_matmul(
      scale_chol,
      scale_chol$transpose(-2L, -1L)
    )
    if (type == "scale") {
      out <- scale
    } else {
      # For the canonical skew-t representation used by the likelihood,
      # delta = alpha / sqrt(1 + alpha^T alpha) and
      # Cov(Y) = C [nu/(nu-2) I - b_nu^2 delta delta^T] C^T,
      # where C is scale_chol and
      # b_nu = sqrt(nu/pi) Gamma((nu-1)/2) / Gamma(nu/2).
      # At nu = Inf, the exact skew-normal limit has scale multiplier one
      # and b_nu^2 = 2 / pi.
      normal <- nu == Inf
      invalid <- nu <= 2
      if (invalid$any()$item()) {
        warning("Component covariance is undefined for nu <= 2; returning NaN for those components.")
      }
      nu_safe <- torch_where(
        normal,
        3 * torch_ones_like(nu),
        nu$clamp(min = 2 + 1e-6)
      )
      alpha_norm_sq <- alpha$pow(2)$sum(dim = -1, keepdim = TRUE)
      delta <- alpha / torch_sqrt(1 + alpha_norm_sq)
      log_b_nu <- 0.5 * (torch_log(nu_safe) -
        torch_log(torch_tensor(pi, device = device, dtype = nu_safe$dtype))) +
        torch_lgamma((nu_safe - 1) / 2) - torch_lgamma(nu_safe / 2)
      b_nu_sq <- torch_where(
        normal,
        (2 / pi) * torch_ones_like(nu_safe),
        torch_exp(2 * log_b_nu)
      )$unsqueeze(-1)$unsqueeze(-1)
      scale_multiplier <- torch_where(
        normal,
        torch_ones_like(nu_safe),
        nu_safe / (nu_safe - 2)
      )$unsqueeze(-1)$unsqueeze(-1)
      eye_mat <- torch_eye(d, device = device, dtype = scale_chol$dtype)$
        unsqueeze(1)$unsqueeze(1)$expand(scale_chol$size())
      delta_outer <- delta$unsqueeze(-1) * delta$unsqueeze(-2)
      standardized_cov <-
        scale_multiplier * eye_mat -
        b_nu_sq * delta_outer
      out <- torch_matmul(
        torch_matmul(scale_chol, standardized_cov),
        scale_chol$transpose(-2L, -1L)
      )
      invalid_expanded <- invalid$unsqueeze(-1)$unsqueeze(-1)$expand(out$size())
      out <- out$masked_fill(invalid_expanded, NaN)
    }
  }
  if (as_array) {
    out <- torch::as_array(out$to(device = "cpu"))
  }
  out
}

################################################################################
