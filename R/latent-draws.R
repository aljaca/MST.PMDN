################################################################################
# Parameter-independent latent draws for MST-PMDN interpretation               #
################################################################################

.validate_seed_mst_pmdn <- function(seed) {
  if (is.null(seed)) {
    return(NULL)
  }
  if (!is.numeric(seed) || length(seed) != 1L || !is.finite(seed) ||
      seed < 0 || seed != floor(seed)) {
    stop("seed must be NULL or a non-negative integer.", call. = FALSE)
  }
  as.integer(seed)
}

# Create a parameter-independent MST-PMDN latent bank
#
# The bank contains uniforms for inverse-CDF component selection, standard
# normals for the skew-normal construction, and uniforms that are transformed
# through the Gamma quantile function for finite Student-t scaling. One latent
# sequence is shared across prediction rows. This deliberately correlates
# Monte Carlo error across rows while leaving every row's marginal predictive
# distribution unchanged, and makes case chunking exactly reproducible.
latent_draws_mst_pmdn <- function(num_samples = 4096L,
                                  output_dim,
                                  dtype = torch_float(),
                                  device = "cpu",
                                  seed = NULL) {
  num_samples <- validate_num_samples(num_samples)
  if (!is.numeric(output_dim) || length(output_dim) != 1L ||
      !is.finite(output_dim) || output_dim < 1 ||
      output_dim != floor(output_dim)) {
    stop("output_dim must be a positive integer.", call. = FALSE)
  }
  output_dim <- as.integer(output_dim)
  seed <- .validate_seed_mst_pmdn(seed)
  if (!is.null(seed)) {
    torch_manual_seed(seed)
  }

  out <- list(
    component_u = torch_rand(
      c(num_samples, 1L), dtype = dtype, device = device
    ),
    skew_z0 = torch_randn(
      c(num_samples, 1L), dtype = dtype, device = device
    ),
    skew_z = torch_randn(
      c(num_samples, output_dim), dtype = dtype, device = device
    ),
    gamma_u = torch_rand(
      c(num_samples, 1L), dtype = dtype, device = device
    ),
    num_samples = num_samples,
    output_dim = output_dim,
    dtype = dtype,
    device = device,
    seed = seed,
    .cache = new.env(parent = emptyenv())
  )
  class(out) <- "mst_pmdn_latent_draws"
  out
}

.validate_latent_draws_mst_pmdn <- function(latent_draws,
                                             output_dim,
                                             dtype,
                                             device) {
  if (!is.list(latent_draws)) {
    stop("latent_draws must be returned by latent_draws_mst_pmdn().",
         call. = FALSE)
  }
  required <- c("component_u", "skew_z0", "skew_z", "gamma_u")
  if (!all(required %in% names(latent_draws)) ||
      !all(vapply(latent_draws[required], inherits, logical(1),
                  what = "torch_tensor"))) {
    stop("latent_draws is missing required torch tensors.", call. = FALSE)
  }

  component_size <- as.integer(latent_draws$component_u$size())
  z0_size <- as.integer(latent_draws$skew_z0$size())
  z_size <- as.integer(latent_draws$skew_z$size())
  gamma_size <- as.integer(latent_draws$gamma_u$size())
  if (length(component_size) != 2L || component_size[2] != 1L ||
      !identical(z0_size, component_size) ||
      !identical(gamma_size, component_size) ||
      length(z_size) != 2L || z_size[1] != component_size[1] ||
      z_size[2] != output_dim) {
    stop(
      paste0(
        "latent_draws must contain component_u, skew_z0, and gamma_u with ",
        "shape [num_samples, 1], and skew_z with shape ",
        "[num_samples, output_dim]."
      ),
      call. = FALSE
    )
  }
  if (component_size[1] < 1L) {
    stop("latent_draws must contain at least one draw.", call. = FALSE)
  }
  for (name in required) {
    tensor <- latent_draws[[name]]
    if (tensor$dtype != dtype) {
      stop(
        sprintf(
          "latent_draws$%s has dtype %s but predictions use %s.",
          name, format(tensor$dtype), format(dtype)
        ),
        call. = FALSE
      )
    }
  }

  # Moving an explicitly supplied bank between devices preserves its values.
  latent_draws[required] <- lapply(
    latent_draws[required],
    function(x) x$to(device = device)
  )
  latent_draws$num_samples <- component_size[1]
  latent_draws$output_dim <- output_dim
  latent_draws$dtype <- dtype
  latent_draws$device <- device
  if (!is.environment(latent_draws$.cache)) {
    latent_draws$.cache <- new.env(parent = emptyenv())
  }
  class(latent_draws) <- "mst_pmdn_latent_draws"
  latent_draws
}

.latent_draws_for_output_mst_pmdn <- function(latent_draws) {
  if (is.null(latent_draws)) return(NULL)
  out <- latent_draws
  # The mutable qgamma cache is an evaluation detail. Keeping it in a public
  # result can add tens of megabytes to saveRDS() output and is unnecessary for
  # exact reproduction because the parameter-independent tensors are retained.
  out$.cache <- NULL
  out
}

.select_components_mst_pmdn <- function(pi, component_u) {
  B <- pi$size(1)
  S <- component_u$size(1)
  cumulative <- pi$cumsum(dim = 2L)$unsqueeze(2L) # [B, 1, M]
  uniforms <- component_u$transpose(1L, 2L)$
    expand(c(B, S))$
    unsqueeze(3L)                                  # [B, S, 1]
  # R torch component indices are 1-based. Counting cumulative probabilities
  # strictly below u and adding one performs inverse-CDF selection directly in
  # that convention.
  ((uniforms > cumulative)$sum(dim = 3L) + 1L)$
    clamp(max = pi$size(2L))$
    to(dtype = torch_long())
}

.uniform_probability_bounds_mst_pmdn <- function(dtype) {
  if (dtype == torch_float()) {
    return(c(lower = 2^-25, upper = 1 - 2^-25))
  }
  # The lower midpoint is half a double-precision uniform quantum. The upper
  # bound uses the largest representable double below one.
  c(lower = 2^-54, upper = 1 - 2^-53)
}

.cache_key_elements_mst_pmdn <- function(key) {
  if (is.null(key)) return(0L)
  sum(vapply(key, length, integer(1)))
}

.gamma_scale_from_uniform_mst_pmdn <- function(nu,
                                                gamma_u,
                                                device,
                                                cache = NULL,
                                                cache_key = NULL,
                                                component_u = NULL) {
  B <- nu$size(1)
  S <- nu$size(2)
  normal <- nu == Inf
  if (normal$all()$item()) {
    return(torch_ones_like(nu))
  }

  nu_safe <- torch_where(normal, 2 * torch_ones_like(nu), nu)
  bounds <- .uniform_probability_bounds_mst_pmdn(gamma_u$dtype)
  gamma_key <- as.numeric(torch::as_array(gamma_u$to(device = "cpu")))
  gamma_key <- pmin(pmax(gamma_key, bounds["lower"]), bounds["upper"])
  component_key <- if (is.null(component_u)) {
    NULL
  } else {
    as.numeric(torch::as_array(component_u$to(device = "cpu")))
  }

  entries <- if (is.environment(cache) &&
                 is.list(cache$gamma_scale_entries)) {
    cache$gamma_scale_entries
  } else {
    list()
  }
  if (is.environment(cache)) {
    same_bank <- identical(cache$gamma_scale_gamma_u, gamma_key) &&
      identical(cache$gamma_scale_component_u, component_key)
    if (!same_bank) {
      entries <- list()
      cache$gamma_scale_entries <- entries
      cache$gamma_scale_gamma_u <- gamma_key
      cache$gamma_scale_component_u <- component_key
    }
  }

  expanded_nu_key <- if (is.null(cache_key)) {
    as.numeric(torch::as_array(nu$to(device = "cpu")))
  } else {
    NULL
  }
  hit <- which(vapply(entries, function(entry) {
    identical(entry$dim, c(B, S)) &&
      (if (is.null(cache_key)) {
        is.null(entry$cache_key) &&
          identical(entry$expanded_nu, expanded_nu_key)
      } else {
        identical(entry$cache_key, cache_key)
      })
  }, logical(1)))[1L]

  if (length(hit) && !is.na(hit)) {
    chi2_r <- entries[[hit]]$chi2
    # Recently used chunks stay resident when the memory limit is reached.
    entry <- entries[[hit]]
    entries <- c(entries[-hit], list(entry))
    cache$gamma_scale_entries <- entries
    cache$gamma_scale_hits <- if (is.null(cache$gamma_scale_hits)) {
      1L
    } else {
      cache$gamma_scale_hits + 1L
    }
  } else {
    u_r <- rep(gamma_key, each = B)
    nu_r <- as.numeric(torch::as_array(nu_safe$to(device = "cpu")))
    chi2_r <- stats::qgamma(u_r, shape = nu_r / 2, scale = 2)
    if (is.environment(cache)) {
      entries[[length(entries) + 1L]] <- list(
        dim = c(B, S),
        cache_key = cache_key,
        expanded_nu = expanded_nu_key,
        chi2 = chi2_r
      )
      # Compact pre-gather pi/nu keys let repeated functionals reuse every
      # prediction chunk. The element budget, rather than a small entry cap,
      # remains the binding memory safeguard.
      bank_elements <- length(gamma_key) + length(component_key)
      while (length(entries) > 64L ||
             (length(entries) > 1L && bank_elements + sum(vapply(
               entries,
               function(entry) {
                 length(entry$expanded_nu) +
                   .cache_key_elements_mst_pmdn(entry$cache_key) +
                   length(entry$chi2)
               },
               integer(1)
             )) > 2e6)) {
        entries <- entries[-1L]
      }
      cache$gamma_scale_entries <- entries
      cache$gamma_scale_misses <- if (is.null(cache$gamma_scale_misses)) {
        1L
      } else {
        cache$gamma_scale_misses + 1L
      }
    }
  }
  chi2 <- torch_tensor(
    matrix(chi2_r, nrow = B, ncol = S),
    dtype = nu$dtype,
    device = device
  )
  finite_scale <- torch_sqrt(nu_safe / chi2$clamp(min = 1e-30))
  torch_where(normal, torch_ones_like(finite_scale), finite_scale)
}

.sample_with_latent_mst_pmdn <- function(pred, latent_draws, device = "cpu") {
  pi <- pred$pi$to(device = device)
  mu <- pred$mu$to(device = device)
  scale_chol <- pred$scale_chol$to(device = device)
  nu <- pred$nu$to(device = device)
  alpha <- pred$alpha$to(device = device)
  skew_none <- .validate_skew_none(pred, "pred")

  B <- pi$size(1)
  d <- mu$size(3)
  S <- latent_draws$num_samples
  idx <- .select_components_mst_pmdn(pi, latent_draws$component_u)
  idx_d <- idx$unsqueeze(3L)$expand(c(B, S, d))
  idx_dd <- idx$unsqueeze(3L)$unsqueeze(4L)$expand(c(B, S, d, d))

  mu_s <- mu$gather(2L, idx_d)
  chol_s <- scale_chol$gather(2L, idx_dd)
  nu_s <- nu$gather(2L, idx)
  gamma_cache_key <- list(
    dim = c(B, S),
    pi_dim = as.integer(pi$size()),
    nu_dim = as.integer(nu$size()),
    pi = as.numeric(torch::as_array(pi$to(device = "cpu"))),
    nu = as.numeric(torch::as_array(nu$to(device = "cpu")))
  )
  W <- .gamma_scale_from_uniform_mst_pmdn(
    nu_s,
    latent_draws$gamma_u,
    device = device,
    cache = latent_draws$.cache,
    cache_key = gamma_cache_key,
    component_u = latent_draws$component_u
  )$unsqueeze(3L)

  z1 <- latent_draws$skew_z$unsqueeze(1L)$expand(c(B, S, d))
  if (skew_none) {
    X <- z1
  } else {
    alpha_s <- alpha$gather(2L, idx_d)
    alpha_norm_sq <- alpha_s$pow(2)$sum(dim = 3L, keepdim = TRUE)
    delta <- alpha_s / torch_sqrt(1 + alpha_norm_sq)
    delta_norm_sq <- delta$pow(2)$sum(dim = 3L, keepdim = TRUE)
    z0 <- latent_draws$skew_z0$unsqueeze(1L)$expand(c(B, S, 1L))
    sqrt_one_minus_delta_sq <- torch_sqrt(
      (1 - delta_norm_sq)$clamp(min = 1e-12)
    )
    delta_dot_z1 <- (delta * z1)$sum(dim = 3L, keepdim = TRUE)
    residual <- z1 - delta * delta_dot_z1 /
      (1 + sqrt_one_minus_delta_sq)
    X <- delta * torch_abs(z0) + residual
  }

  samples <- mu_s + W * torch_matmul(chol_s, X$unsqueeze(4L))$squeeze(4L)
  list(
    samples = samples$permute(c(2L, 1L, 3L)),
    components = idx$permute(c(2L, 1L))
  )
}

print.mst_pmdn_latent_draws <- function(x, ...) {
  cat(
    "MST-PMDN latent bank:\n",
    "  draws: ", x$num_samples, "\n",
    "  output dimensions: ", x$output_dim, "\n",
    "  dtype: ", format(x$dtype), "\n",
    "  device: ", format(x$component_u$device), "\n",
    sep = ""
  )
  invisible(x)
}
