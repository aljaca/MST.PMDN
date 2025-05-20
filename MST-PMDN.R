################################################################################
# Deep Multivariate skew t-Parsimonious Mixture Density Network (MST-PMDN)     #
# Alex J. Cannon <alex.cannon@ec.gc.ca>                                        #
################################################################################

library(torch)

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

build_orthogonal_matrix <- function(params, dim) {
  # Helper for building an orthogonal orientation matrix
  batch_size <- params$size(1)
  X <- torch_zeros(batch_size, dim, dim)
  indices <- torch_triu_indices(dim, dim, offset = 1)
  row_indices <- indices[1, ]$add(1)$to(dtype = torch_long())
  col_indices <- indices[2, ]$add(1)$to(dtype = torch_long())
  batch_indices <- torch_arange(1, batch_size)$unsqueeze(2)$expand(
    c(batch_size, indices$size(2)))$to(dtype = torch_long())
  row_indices <- row_indices$unsqueeze(1)$expand(c(batch_size,
                                                   -1))$to(dtype = torch_long())
  col_indices <- col_indices$unsqueeze(1)$expand(c(batch_size,
                                                   -1))$to(dtype = torch_long())
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
    ## mu is a parameter
    with_no_grad({
      model$mu$copy_(cent)
    })
  } else {
    ## mu comes from bias of fc_mu
    with_no_grad({
      model$fc_mu$bias$copy_(cent$reshape(c(-1)))
      model$fc_mu$g$zero_()
    })
  }
}

# ------------------------------------------------
# Student-t CDF functions (waiting for torch for R
# implementation of torch.distributions.studentT)
# ------------------------------------------------

t_pdf_int <- function(x, nu, pi_const) {
  # Student-t PDF for t_cdf_int
  nu_for_calc <- nu
  if (nu$dim() > 0) {
    if (x$dim() > nu$dim() && (x$dim() == nu$dim() + 1)) {
      nu_for_calc <- nu$unsqueeze(-1L)
    }
  }
  # Log of Gamma functions
  log_gamma_nu_plus_1_div_2 <- torch_lgamma((nu_for_calc + 1.0) / 2.0)
  log_gamma_nu_div_2 <- torch_lgamma(nu_for_calc / 2.0)
  # Coefficient part
  coeff_num <- torch_exp(log_gamma_nu_plus_1_div_2)
  coeff_den_sqrt_term <- torch_sqrt(nu_for_calc * pi_const)
  coeff_den <- coeff_den_sqrt_term * torch_exp(log_gamma_nu_div_2)
  coeff <- coeff_num / coeff_den
  # Main term: (1 + x^2/nu)^(-(nu+1)/2)
  # x_squared_div_nu term: x^2 / nu_for_calc
  x_squared_div_nu <- torch_pow(x, 2L) / nu_for_calc
  base <- 1.0 + x_squared_div_nu
  # exponent term: -(nu_for_calc + 1.0) / 2.0
  exponent <- -(nu_for_calc + 1.0) / 2.0
  main_term <- torch_pow(base, exponent)
  return(coeff * main_term)
}

t_cdf_int <- function(t_val, nu, num_integration_points = 1000L) {
  # Student-t CDF (slow approximation via integration of PDF)
  if (!inherits(t_val, "torch_tensor")) {
    stop("t_val must be a torch_tensor.")
  }
  original_t_dtype <- t_val$dtype
  if (!inherits(nu, "torch_tensor")) {
    nu <- torch_scalar_tensor(nu, dtype = t_val$dtype, device = t_val$device)
  }
  original_nu_dtype <- nu$dtype
  if ((nu <= 0)$any()$item()) {
    stop("All elements of degrees of freedom 'nu' must be positive.")
  }
  if (!t_val$dtype$is_floating_point) {
    t_val <- t_val$to(dtype = torch_float())
  }
  if (!nu$dtype$is_floating_point) {
    nu <- nu$to(dtype = torch_float()) 
  }
  promoted_dtype <- torch_promote_types(t_val$dtype, nu$dtype)
  if (!promoted_dtype$is_floating_point) { 
      promoted_dtype <- torch_float()
  }
  if (t_val$dtype != promoted_dtype) {
    t_val <- t_val$to(dtype = promoted_dtype)
  }
  if (nu$dtype != promoted_dtype) {
    nu <- nu$to(dtype = promoted_dtype)
  }
  pi_const <- torch_scalar_tensor(3.14159265359, dtype = promoted_dtype,
                                  device = t_val$device)
  # Integral calculation
  abs_t <- torch_abs(t_val)  
  abs_t_unsqueezed <- abs_t$unsqueeze(-1L)
  base_integration_domain <- torch_linspace(0, 1, num_integration_points, 
                                            dtype = promoted_dtype,
                                            device = t_val$device)
  integration_x <- abs_t_unsqueezed * base_integration_domain
  integration_y <- t_pdf_int(integration_x, nu, pi_const)
  # Integrate using trapezoidal rule along the last dimension (the integration
  # points dimension). Resulting shape will be *original_t_shape
  integral_0_to_abs_t <- torch_trapz(integration_y, x = integration_x,
                                     dim = -1L)
  # CDF calculation
  sign_t <- torch_sign(t_val)
  half_const <- torch_scalar_tensor(0.5, dtype = promoted_dtype,
                                    device = t_val$device)
  cdf_val <- half_const + sign_t * integral_0_to_abs_t  
  return(cdf_val)
}

t_cdf_slow <- function(z, nu) {
  # Student-t CDF (slow R pt and finite difference for nu gradient)
  # Allows gradients to flow through computational graph
  # Store original shape and device
  z_shape <- z$size()
  z_device <- z$device
  z_flat <- z$reshape(-1)
  batch_size <- z_flat$size(1)
  if (inherits(nu, "torch_tensor")) {
    if (nu$dim() == 0) {
      nu_flat <- nu$expand(batch_size)
    } else {
      nu_flat <- nu$reshape(-1)
      if (nu_flat$size(1) == 1 && batch_size > 1) {
        nu_flat <- nu_flat$expand(batch_size)
      }
    }
    nu_device <- nu$device
  } else {
    nu_flat <- torch_tensor(rep(nu, batch_size), dtype = z$dtype,
                            device = z_device)
    nu_device <- z_device
  }
  # Move to CPU for R functions (pt/dt)
  if (as.character(z_device) == "cpu") {
    z_cpu <- z_flat
    nu_cpu <- nu_flat
  } else {
    z_cpu <- z_flat$to(device = "cpu")
    nu_cpu <- nu_flat$to(device = "cpu")
  }
  # Calculate CDF values using R
  z_vals <- as.numeric(z_cpu)
  nu_vals <- as.numeric(nu_cpu)
  cdf_vals <- pt(z_vals, df = nu_vals)
  # Convert back to tensor on original device
  cdf_flat <- torch_tensor(cdf_vals, dtype = z$dtype, device = z_device)
  # Gradient logic (surrogate gradients)
  if (z$requires_grad || (inherits(nu, "torch_tensor") && nu$requires_grad)) {
    # PDF values for z gradient
    if (z$requires_grad) {
      pdf_vals <- dt(z_vals, df = nu_vals)
      pdf_flat <- torch_tensor(pdf_vals, dtype = z$dtype, device = z_device)
    }
    # Finite difference for nu gradient
    if (inherits(nu, "torch_tensor") && nu$requires_grad) {
      delta <- 0.01
      nu_plus_vals <- nu_vals + delta
      nu_minus_vals <- pmax(nu_vals - delta, 0.01)
      cdf_plus_vals <- pt(z_vals, df = nu_plus_vals)
      cdf_minus_vals <- pt(z_vals, df = nu_minus_vals)
      d_cdf_d_nu_vals <- (cdf_plus_vals - cdf_minus_vals) / (2 * delta)
      d_cdf_d_nu_flat <- torch_tensor(d_cdf_d_nu_vals, dtype = nu$dtype,
                                      device = nu_device)
    }
    # Reshape to original dimensions
    cdf <- cdf_flat$reshape(z_shape)
    result <- cdf
    if (z$requires_grad) {
      pdf <- pdf_flat$reshape(z_shape)
      result <- result + pdf * (z - z$detach())
    }
    if (inherits(nu, "torch_tensor") && nu$requires_grad) {
      d_cdf_d_nu <- d_cdf_d_nu_flat$reshape(z_shape)
      nu_expanded <- nu$expand(z_shape)
      result <- result + d_cdf_d_nu * (nu_expanded - nu_expanded$detach())
    }
    return(result)
  }
  # For non-gradient case, reshape and return
  cdf_flat$reshape(z_shape)
}

t_cdf_fast <- function(z, nu) {
  # Student-t CDF (fast scaled normal approximation for large values of nu)
  nu_f <- nu$to(dtype = z$dtype)
  s    <- torch_sqrt(nu_f / (nu_f - torch_tensor(2, dtype = z$dtype,
                                                 device = z$device)))
  distr_normal(0, 1)$cdf(s * z)
}

t_cdf <- function(z, nu, nu_switch = 20) {
  # Switches between slow and fast Student-t CDF implementations when
  # nu >= nu_switch (swap t_cdf_slow for t_cdf_int if working on GPU)
  torch_where(nu >= nu_switch, t_cdf_fast(z, nu), t_cdf_slow(z, nu))
}

log_pt <- function(z, nu, nu_switch = 20) {
  torch_log(torch_clamp(t_cdf(z, nu, nu_switch), min = 1e-12))
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
  fixed_nu = NULL,
  range_nu = c(3., 50.),    # clamp nu range
  max_alpha = 5.,           # alpha = [-max_alpha, max_alpha]
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
      self$hidden_dims     <- as.integer(hidden_dim)
      self$n_mixtures      <- n_mixtures
      self$output_dim      <- output_dim
      self$constraint      <- constraint
      self$constant_attr   <- constant_attr
      self$min_nu          <- min(range_nu)
      self$max_nu          <- max(range_nu)
      self$max_alpha       <- max_alpha
      self$min_vol_shape   <- min_vol_shape
      self$min_mix_weight  <- min_mix_weight
      self$jitter          <- jitter
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
      # --------------------
      # Mixture weights (pi)
      # --------------------
      if (grepl("x", constant_attr)) {
        self$pi <- nn_parameter(torch_ones(n_mixtures) / n_mixtures)
      } else {
        self$fc_pi <- weight_norm_linear(self$final_hidden_dim, n_mixtures)
      }
      # ----------
      # Means (mu)
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
      self$nu_normal_approx <- (nu_letter == "N")  # means treat as ~Inf
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
            # Initialize weights and scales to zero for stability
            self$fc_nu_partial$V$detach()$fill_(0)
            self$fc_nu_partial$g$detach()$fill_(0)
            self$fc_nu_partial$bias$detach()$fill_(0)
          }
        }
      } else if (!self$nu_normal_approx) {
        nu_size <- if (self$nu_shared) 1 else n_mixtures
        if (grepl("n", constant_attr)) {
          # i.e., nu is "constant" (but distinct across mixture if not shared)
          # Initialize raw_nu to 0 (targets middle of range via sigmoid)
          self$nu_param <- nn_parameter(torch_zeros(nu_size))
        } else {
          # Define a weight-normalized linear layer & custom initialize it
          self$fc_nu <- weight_norm_linear(self$final_hidden_dim, nu_size)
          # Initialize raw_nu bias to 0 (targets middle of range via sigmoid)
          # Initialize weights and scales to zero for stability
          self$fc_nu$V$detach()$fill_(0)    # zero all weights
          self$fc_nu$g$detach()$fill_(0)    # zero all scale factors
          self$fc_nu$bias$detach()$fill_(0) # Bias=0 -> sigmoid=0.5
        }
      }
      # ----------------
      # Skewness (alpha)
      # ----------------
      skew_letter <- substr(constraint, 5, 5)
      self$skew_none   <- (skew_letter == "N")
      self$skew_shared <- (skew_letter == "E")
      self$skew_vary   <- (!self$skew_none && !self$skew_shared)
      if (!self$skew_none) {
        alpha_size <- if (self$skew_shared) output_dim else (n_mixtures *
                                                               output_dim)
        if (grepl("s", constant_attr)) {
          self$alpha_param <- nn_parameter(torch_zeros(alpha_size))
        } else {
          self$fc_alpha <- weight_norm_linear(self$final_hidden_dim, alpha_size)
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
        # Concatenate features from both branches
        combined_features <- torch_cat(list(tabular_features, image_features),
                                       dim = 2)
      } else {
        # Only tabular features
        combined_features <- tabular_features
      }
      # Continue with existing pipeline using combined features
      h <- self$hidden(combined_features)
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
      # Means (mu)
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
        # Clamp logits and use soft‑plus(+ε) to prevent under/overflow
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
        # Learned shape: same clamp‑softplus safeguard
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
      # Final safety clamp keeps Σ well‑conditioned
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
      if (self$nu_normal_approx) {
        # effectively infinite => normal. Use max_nu as a large proxy.
        nu <- torch_full(c(B, self$n_mixtures), self$max_nu, device = x$device)
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

loss_mst_pmdn <- function(output, target, nu_switch = 20) {
  # Output must have: pi, mu, scale (Cholesky L), nu, alpha
  # target shape: [B, d]
  pi         <- output$pi         # [B, M]
  mu         <- output$mu         # [B, M, d]
  scale_chol <- output$scale_chol # [B, M, d, d]
  nu         <- output$nu         # [B, M]
  alpha      <- output$alpha      # [B, M, d]
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
  # Log PDF of multivariate t distribution component
  half_nu <- nu / 2
  half_nu_plus_d <- (nu + d) / 2
  lg_nu_div2 <- torch_lgamma(half_nu)
  lg_nuplusd_div2 <- torch_lgamma(half_nu_plus_d)
  logC_t <- lg_nuplusd_div2 - lg_nu_div2 - (d / 2) * torch_log(nu *
                torch_tensor(3.14159265359, device = dev)) - 0.5 * log_det_Sigma
  logTail <- -half_nu_plus_d * torch_log1p(torch_clamp(maha / nu,
                                           min = -1 + 1e-7, max = 1e7))
  log_pdf_t <- logC_t + logTail
  # Skewness factor calculation
  # Clamp large values, [B, M, 1]
  cterm <- torch_sqrt((nu + d) / (nu + maha))$unsqueeze(-1)$clamp(max = 1e6)
  w <- cterm * v # [B, M, d]
  # alpha^T w
  alpha_dot_w <- (alpha * w)$sum(dim = 3) # [B, M]
  # Univariate standard t-CDF with df = nu + d
  log_skew_factor <- torch_log(2.0) + log_pt(alpha_dot_w, nu + d, nu_switch)
  # Final log-density of skew-t component
  log_skewt <- log_pdf_t + log_skew_factor # [B, M]
  # Mixture weighting and log-sum-exp for total log-likelihood
  # log P(y|x) = log sum_k [ pi_k * SkewT(y | mu_k, Sigma_k, alpha_k, nu_k) ]
  #            = logsumexp_k [ log(pi_k) + log(SkewT(...)) ]
  weighted_log_probs <- torch_log(pi$clamp(min = 1e-12)) + log_skewt # [B, M]
  # Negative log-likelihood (average over batch)
  loss <- -torch_logsumexp(weighted_log_probs, dim = 2)$mean()
  loss
}

# -----------------------------------------------
# Skew t-distribution random sampling (on device)
# *_df version converts to R data frame
# -----------------------------------------------

sample_mst_pmdn <- function(mdn_output, num_samples = 1, device = "cpu") {
  # gather parameters
  pi     <- mdn_output$pi          $to(device = device)
  mu     <- mdn_output$mu          $to(device = device)
  L_all  <- mdn_output$scale_chol  $to(device = device)
  nu_all <- mdn_output$nu          $to(device = device)
  alpha_all <- mdn_output$alpha    $to(device = device)
  B <- pi$size(1)
  # M <- pi$size(2)
  d <- mu$size(3)
  # component indices (1-based)
  idx      <- pi$multinomial(num_samples, replacement = TRUE)$add(1L)
  idx_d    <- idx$unsqueeze(-1)$expand(c(B, num_samples, d))
  idx_dd   <- idx$unsqueeze(-1)$unsqueeze(-1)$expand(c(B, num_samples, d, d))
  # gather parameters for the selected components
  mu_s    <- mu       $gather(2, idx_d)
  L_s     <- L_all    $gather(2, idx_dd)
  nu_s    <- nu_all   $gather(2, idx)
  alpha_s <- alpha_all$gather(2, idx_d)
  # Gamma scaling for Student-t tails
  chi2 <- sample_gamma(nu_s / 2, scale = 2, device = device)
  W    <- torch_sqrt(nu_s / chi2$clamp(min = 1e-12))$unsqueeze(-1)
  # skew direction (identity‑covariance, Sigma = I convention)
  alpha_norm_sq <- alpha_s$pow(2)$sum(dim = -1, keepdim = TRUE)
  delta <- alpha_s / torch_sqrt(1 + alpha_norm_sq + 1e-10)
  delta_norm_sq <- delta$pow(2)$sum(dim = -1, keepdim = TRUE)
  # standard normals
  z0 <- torch_randn(c(B, num_samples, 1), device = device)
  z1 <- torch_randn(c(B, num_samples, d), device = device)
  # skew‑normal core
  X <- delta * torch_abs(z0) +
    torch_sqrt((1 - delta_norm_sq)$clamp(min = 1e-12)) * z1
  # affine map to response space  Y
  Y <- mu_s + W * (torch_matmul(L_s, X$unsqueeze(-1))$squeeze(-1))
  # return both samples and component IDs
  list(
    samples    = Y$permute(c(2, 1, 3)),
    components = idx$permute(c(2, 1))
  )
}

sample_mst_pmdn_df <- function(mdn_output, num_samples = 1, device = "cpu") {
  # gather parameters
  pi         <- mdn_output$pi         $to(device = device)
  mu         <- mdn_output$mu         $to(device = device)
  L_all      <- mdn_output$scale_chol $to(device = device)
  nu_all     <- mdn_output$nu         $to(device = device)
  alpha_all  <- mdn_output$alpha      $to(device = device)
  B <- pi$size(1)
  # M <- pi$size(2)
  d <- mu$size(3)
  # component indices (1‑based)
  idx    <- pi$multinomial(num_samples, replacement = TRUE)$add(1L)
  idx_d  <- idx$unsqueeze(-1)$expand(c(B, num_samples, d))
  idx_dd <- idx$unsqueeze(-1)$unsqueeze(-1)$expand(c(B, num_samples, d, d))
  # gather parameters for the selected components
  mu_s     <- mu        $gather(2, idx_d)
  L_s      <- L_all     $gather(2, idx_dd)
  nu_s     <- nu_all    $gather(2, idx)
  alpha_s  <- alpha_all $gather(2, idx_d)
  # Gamma scaling for Student‑t tails
  chi2 <- sample_gamma(nu_s / 2, scale = 2, device = device)
  W    <- torch_sqrt(nu_s / chi2$clamp(min = 1e-12))$unsqueeze(-1)
  # skew direction (identity‑covariance, Sigma = I convention)
  alpha_norm_sq <- alpha_s$pow(2)$sum(dim = -1, keepdim = TRUE)
  delta         <- alpha_s / torch_sqrt(1 + alpha_norm_sq)
  delta_norm_sq <- delta$pow(2)$sum(dim = -1, keepdim = TRUE)
  # standard normals
  z0 <- torch_randn(c(B, num_samples, 1), device = device)
  z1 <- torch_randn(c(B, num_samples, d), device = device)
  # skew‑normal core
  X <- delta * torch_abs(z0) +
    torch_sqrt((1 - delta_norm_sq)$clamp(min = 1e-12)) * z1
  # affine map to response space  Y
  Y <- mu_s + W * (torch_matmul(L_s, X$unsqueeze(-1))$squeeze(-1))
  # reshape to long data‑frame
  S   <- num_samples
  mat  <- as.matrix(Y$reshape(c(B * S, d))$cpu())
  comp <- as.integer(idx$reshape(c(B * S))$cpu())
  data.frame(mat,
             row  = rep(seq_len(B), each = S),
             draw = rep(seq_len(S), times = B),
             comp = factor(comp))
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
                           nu_switch = 20,
                           max_alpha = 5.,
                           min_vol_shape = 1e-2,
                           min_mix_weight = 1e-4,
                           jitter = 1e-6,
                           activation = nn_tanh,
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
                           device = "cpu"
) {
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
      image_inputs <- torch_tensor(image_inputs, device = device, dtype = torch_float())
    else
      image_inputs <- image_inputs$to(device = device)
  }
  # Data training/validation split
  n_total <- inputs$size(1)
  if (!is.null(custom_split)) {
    if (is.list(custom_split) && all(c("train", "validation") %in% names(custom_split))) {
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
    } else stop("Invalid custom_split format.")
    if (length(train_indices) == 0 || length(val_indices) == 0)
      stop("Both training and validation sets must contain at least one sample")
  } else if (validation_split > 0 && validation_split < 1) {
    val_indices <- sample.int(n_total, size = floor(n_total * validation_split))
    train_indices <- setdiff(seq_len(n_total), val_indices)
  } else {
    train_indices <- seq_len(n_total)
    val_indices <- integer(0)
  }
  train_inputs <- inputs[train_indices, ]
  train_outputs <- outputs[train_indices, ]
  train_image_inputs <- if (!is.null(image_inputs)) image_inputs[train_indices, ] else NULL
  if (length(val_indices) > 0) {
    val_inputs <- inputs[val_indices, ]
    val_outputs <- outputs[val_indices, ]
    val_image_inputs <- if (!is.null(image_inputs)) image_inputs[val_indices, ] else NULL
  } else {
    val_inputs <- NULL
    val_outputs <- NULL
    val_image_inputs <- NULL
  }
  # Model initialization logic
  checkpoint <- NULL
  if (!is.null(model)) {
    # Use provided model, reset all counters/optimizer
    model <- model$to(device = device)
    start_epoch <- 1
    train_loss_history <- rep(NA_real_, epochs)
    val_loss_history   <- rep(NA_real_, epochs)
    best_val_loss     <- Inf
    best_val_epoch    <- NA
    no_improve_epochs <- 0
    best_train_loss   <- Inf
    best_train_epoch  <- NA
  } else if (resume_from_checkpoint && file.exists(checkpoint_path)) {
    # Load from checkpoint, restore everything
    checkpoint <- torch_load(checkpoint_path)
    input_dim  <- inputs$size(2)
    output_dim <- outputs$size(2)
    model <- define_mst_pmdn(
      input_dim, output_dim, hidden_dim, n_mixtures,
      constraint, constant_attr,
      activation = activation,
      drop_hidden = drop_hidden,
      image_module = image_module,
      tabular_module = tabular_module,
      fixed_nu = fixed_nu,
      range_nu = range_nu,
      max_alpha = max_alpha,
      min_vol_shape = min_vol_shape,
      min_mix_weight = min_mix_weight,
      jitter = jitter
    )
    model$load_state_dict(checkpoint$model_state_dict)
    model <- model$to(device = device)
    train_loss_history <- checkpoint$train_loss_history
    val_loss_history   <- checkpoint$val_loss_history
    start_epoch        <- checkpoint$epoch + 1
    best_val_loss      <- checkpoint$best_val_loss
    best_val_epoch     <- checkpoint$best_val_epoch
    no_improve_epochs  <- checkpoint$no_improve_epochs
    best_train_loss    <- if (!is.null(checkpoint$best_train_loss)) checkpoint$best_train_loss else Inf
    best_train_epoch   <- if (!is.null(checkpoint$best_train_epoch)) checkpoint$best_train_epoch else NA
    cat(sprintf("Resumed from checkpoint at epoch %d with best_val_loss=%.4f\n", checkpoint$epoch, best_val_loss))
  } else {
    # New model, new optimizer, reset everything
    input_dim  <- inputs$size(2)
    output_dim <- outputs$size(2)
    model <- define_mst_pmdn(
      input_dim, output_dim, hidden_dim, n_mixtures,
      constraint, constant_attr,
      activation = activation,
      drop_hidden = drop_hidden,
      image_module = image_module,
      tabular_module = tabular_module,
      fixed_nu = fixed_nu,
      range_nu = range_nu,
      max_alpha = max_alpha,
      min_vol_shape = min_vol_shape,
      min_mix_weight = min_mix_weight,
      jitter = jitter
    )
    model$apply(init_weight_norm)
    model <- model$to(device = device)
    init_mu_kmeans(model, outputs_train = train_outputs, n_mixtures = n_mixtures,
                   constant_attr = constant_attr, device = device)
    start_epoch <- 1
    train_loss_history <- rep(NA_real_, epochs)
    val_loss_history   <- rep(NA_real_, epochs)
    best_val_loss     <- Inf
    best_val_epoch    <- NA
    no_improve_epochs <- 0
    best_train_loss   <- Inf
    best_train_epoch  <- NA
  }
  # Adam optimizer
  img_params    <- if (!is.null(model$image_module))   model$image_module$parameters   else list()
  tab_params    <- if (!is.null(model$tabular_module)) model$tabular_module$parameters else list()
  hidden_params <- model$hidden$parameters
  all_params    <- model$parameters
  feat_params   <- c(img_params, tab_params, hidden_params)
  head_params   <- setdiff(all_params, feat_params)
  optimizer <- optim_adam(
    params = list(
      list(params = img_params, weight_decay = wd_image),
      list(params = tab_params, weight_decay = wd_tabular),
      list(params = hidden_params),
      list(params = head_params)
    ),
    lr = lr
  )
  # Restore optimizer state if resuming from checkpoint (but not for new
  # phase/fine-tune)
  if (!is.null(checkpoint) && is.null(model)) {
    optimizer$load_state_dict(checkpoint$optimizer_state_dict)
  }

  best_train_checkpoint_path <- sub("\\.pt$", "_trainbest.pt", checkpoint_path)
  if (identical(best_train_checkpoint_path, checkpoint_path))
    best_train_checkpoint_path <- paste0(checkpoint_path, "_trainbest")
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
        initialize = function(x, im, y) { self$x <- x; self$im <- im; self$y <- y },
        .getitem = function(idx) list(self$x[idx, ], self$im[idx, ], self$y[idx, ]),
        .length = function() self$x$size(1)
      )(inp, img_inp, outp)
    }
  }
  train_dataset <- dataset_fn(train_inputs, train_image_inputs, train_outputs)
  train_loader  <- dataloader(train_dataset, batch_size = batch_size, shuffle = TRUE, drop_last = TRUE)
  if (!is.null(val_inputs)) {
    val_dataset <- dataset_fn(val_inputs, val_image_inputs, val_outputs)
    val_loader  <- dataloader(val_dataset, batch_size = batch_size, shuffle = FALSE, drop_last = TRUE)
  }
  # Training loop
  final_epoch <- NA
  for (epoch in seq.int(start_epoch, epochs)) {
    model$train()
    total_loss  <- 0
    batch_count <- 0
    coro::loop(for (batch in train_loader) {
      optimizer$zero_grad()
      if (length(batch) == 3) {
        inputs_batch       <- batch[[1]]
        image_inputs_batch <- batch[[2]]
        outputs_batch      <- batch[[3]]
        pred <- model(inputs_batch, image_inputs_batch)
      } else {
        inputs_batch  <- batch[[1]]
        outputs_batch <- batch[[2]]
        pred <- model(inputs_batch)
      }
      loss <- loss_mst_pmdn(pred, outputs_batch, nu_switch = nu_switch)
      loss$backward()
      if (!is.null(max_norm)) nn_utils_clip_grad_norm_(model$parameters, max_norm)
      optimizer$step()
      total_loss  <- total_loss + loss$item()
      batch_count <- batch_count + 1
    })
    avg_train_loss <- total_loss / batch_count
    train_loss_history[epoch] <- avg_train_loss
    # Validation
    if (!is.null(val_inputs)) {
      model$eval()
      total_val_loss <- 0
      val_batches    <- 0
      with_no_grad({
        coro::loop(for (batch in val_loader) {
          if (length(batch) == 3) {
            inputs_batch       <- batch[[1]]
            image_inputs_batch <- batch[[2]]
            outputs_batch      <- batch[[3]]
            pred <- model(inputs_batch, image_inputs_batch)
          } else {
            inputs_batch  <- batch[[1]]
            outputs_batch <- batch[[2]]
            pred <- model(inputs_batch)
          }
          loss <- loss_mst_pmdn(pred, outputs_batch, nu_switch = nu_switch)
          total_val_loss <- total_val_loss + loss$item()
          val_batches    <- val_batches + 1
        })
      })
      avg_val_loss <- total_val_loss / val_batches
      val_loss_history[epoch] <- avg_val_loss
      cat(sprintf("Epoch %d - Train Loss: %.4f - Val Loss: %.4f\n", epoch, avg_train_loss, avg_val_loss))
      # Early stopping logic
      if (avg_val_loss < best_val_loss) {
        best_val_loss     <- avg_val_loss
        best_val_epoch    <- epoch
        no_improve_epochs <- 0
        checkpoint <- list(
          epoch                = epoch,
          model_state_dict     = model$state_dict(),
          optimizer_state_dict = optimizer$state_dict(),
          best_val_loss        = best_val_loss,
          best_val_epoch       = best_val_epoch,
          no_improve_epochs    = no_improve_epochs,
          train_loss_history   = train_loss_history,
          val_loss_history     = val_loss_history
        )
        torch_save(checkpoint, checkpoint_path)
        cat(sprintf("Checkpoint saved at epoch %d (best_val_loss=%.4f)\n", epoch, best_val_loss))
      } else {
        no_improve_epochs <- no_improve_epochs + 1
      }
      if (no_improve_epochs >= early_stopping_patience) {
        cat(sprintf("Early stopping triggered at epoch %d (no improvement for %d epochs).\n", epoch, early_stopping_patience))
        final_epoch <- epoch
        break
      }
      # Periodic checkpoint
      if (epoch %% checkpoint_interval == 0) {
        checkpoint <- list(
          epoch                = epoch,
          model_state_dict     = model$state_dict(),
          optimizer_state_dict = optimizer$state_dict(),
          best_val_loss        = best_val_loss,
          best_val_epoch       = best_val_epoch,
          no_improve_epochs    = no_improve_epochs,
          train_loss_history   = train_loss_history,
          val_loss_history     = val_loss_history
        )
        torch_save(checkpoint, checkpoint_path)
        cat(sprintf("Periodic checkpoint saved at epoch %d.\n", epoch))
      }
      # Scheduler step
      if (!is.null(scheduler_step) && (epoch %% scheduler_step == 0)) {
        for (group in optimizer$param_groups) group$lr <- group$lr * scheduler_gamma
        cat(sprintf("Learning rate updated at epoch %d.\n", epoch))
      }
    } else {
      cat(sprintf("Epoch %d - Train Loss: %.4f\n", epoch, avg_train_loss))
      if (avg_train_loss < best_train_loss) {
        best_train_loss  <- avg_train_loss
        best_train_epoch <- epoch
        checkpoint <- list(
          epoch                = epoch,
          model_state_dict     = model$state_dict(),
          optimizer_state_dict = optimizer$state_dict(),
          best_train_loss      = best_train_loss,
          best_train_epoch     = best_train_epoch,
          train_loss_history   = train_loss_history
        )
        torch_save(checkpoint, best_train_checkpoint_path)
        cat(sprintf("Best training checkpoint saved at epoch %d (loss=%.4f)\n", epoch, best_train_loss))
      }
      if (epoch %% checkpoint_interval == 0) {
        checkpoint <- list(
          epoch                = epoch,
          model_state_dict     = model$state_dict(),
          optimizer_state_dict = optimizer$state_dict(),
          best_train_loss      = best_train_loss,
          best_train_epoch     = best_train_epoch,
          train_loss_history   = train_loss_history
        )
        torch_save(checkpoint, checkpoint_path)
        cat(sprintf("Periodic checkpoint saved at epoch %d.\n", epoch))
      }
      if (!is.null(scheduler_step) && (epoch %% scheduler_step == 0)) {
        for (group in optimizer$param_groups) group$lr <- group$lr * scheduler_gamma
        cat(sprintf("Learning rate updated at epoch %d.\n", epoch))
      }
    }
  } # End epoch loop
  if (is.na(final_epoch)) final_epoch <- if (exists("epoch")) epoch else epochs
  # Reload best checkpoint
  if (!is.null(val_inputs) && file.exists(checkpoint_path)) {
    checkpoint <- torch_load(checkpoint_path)
    model$load_state_dict(checkpoint$model_state_dict)
    cat("Best model loaded from validation-based checkpoint.\n")
    best_val_loss     <- checkpoint$best_val_loss
    best_val_epoch    <- checkpoint$best_val_epoch
  } else if (is.null(val_inputs) && file.exists(best_train_checkpoint_path)) {
    checkpoint <- torch_load(best_train_checkpoint_path)
    model$load_state_dict(checkpoint$model_state_dict)
    cat("Best model loaded from training-based checkpoint.\n")
    best_train_loss  <- checkpoint$best_train_loss
    best_train_epoch <- checkpoint$best_train_epoch
  }
  list(
    model              = model,
    train_loss_history = train_loss_history[1:final_epoch],
    val_loss_history   = if (!is.null(val_inputs)) val_loss_history[1:final_epoch] else NULL,
    best_val_epoch     = best_val_epoch,
    best_val_loss      = if (!is.null(val_inputs)) best_val_loss else NULL,
    best_train_epoch   = if (is.null(val_inputs)) best_train_epoch else NULL,
    best_train_loss    = if (is.null(val_inputs)) best_train_loss else NULL,
    final_epoch        = final_epoch,
    train_indices      = train_indices,
    val_indices        = val_indices
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

################################################################################
