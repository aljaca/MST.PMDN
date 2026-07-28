# Benchmark the structural skew-none fast paths without adding timing
# assertions to the test suite. Run from the package root with:
#
#   Rscript tools/benchmark-skew-none-fast-path.R
#
# The recursive configuration intentionally repeats B = S = 1 sampling
# 36,200 times. Override that count only for a smoke run with the
# MST_PMDN_BENCH_RECURSIVE_REPS environment variable.

if (!requireNamespace("torch", quietly = TRUE)) {
  stop("The torch package is required.")
}
if (!requireNamespace("MST.PMDN", quietly = TRUE)) {
  if (!requireNamespace("pkgload", quietly = TRUE)) {
    stop("Install MST.PMDN or pkgload before running this script.")
  }
  pkgload::load_all(".", quiet = TRUE)
} else {
  suppressPackageStartupMessages(library(MST.PMDN))
}

recursive_repetitions <- as.integer(Sys.getenv(
  "MST_PMDN_BENCH_RECURSIVE_REPS", "36200"
))
if (is.na(recursive_repetitions) || recursive_repetitions < 1L) {
  stop("MST_PMDN_BENCH_RECURSIVE_REPS must be a positive integer.")
}

available_devices <- "cpu"
if (torch::cuda_is_available()) {
  available_devices <- c(available_devices, "cuda")
}

sync_device <- function(device) {
  if (identical(device, "cuda")) {
    torch::cuda_synchronize()
  }
}

make_output <- function(B, M, d, nu_value, alpha_value, skew_none, device) {
  dtype <- torch::torch_float()
  pi <- torch::torch_full(
    c(B, M), 1 / M, dtype = dtype, device = device
  )
  mu <- torch::torch_zeros(c(B, M, d), dtype = dtype, device = device)
  scale_chol <- torch::torch_eye(d, dtype = dtype, device = device)$
    reshape(c(1, 1, d, d))$expand(c(B, M, d, d))$clone()
  nu <- torch::torch_full(
    c(B, M), nu_value, dtype = dtype, device = device
  )
  alpha <- torch::torch_full(
    c(B, M, d), alpha_value, dtype = dtype, device = device
  )
  list(
    pi = pi,
    mu = mu,
    scale_chol = scale_chol,
    nu = nu,
    alpha = alpha,
    skew_none = skew_none
  )
}

allocation_count <- function(fun, repetitions) {
  if (!isTRUE(capabilities("profmem"))) {
    return(NA_integer_)
  }
  profile <- tempfile(fileext = ".out")
  on.exit(unlink(profile), add = TRUE)
  Rprofmem(profile)
  on.exit(Rprofmem(NULL), add = TRUE)
  for (i in seq_len(repetitions)) {
    fun()
  }
  Rprofmem(NULL)
  length(readLines(profile, warn = FALSE))
}

benchmark_call <- function(fun, repetitions, device) {
  invisible(gc())
  sync_device(device)
  elapsed <- system.time({
    for (i in seq_len(repetitions)) {
      fun()
    }
    sync_device(device)
  })[["elapsed"]]
  allocation_repetitions <- min(repetitions, 100L)
  allocations <- allocation_count(fun, allocation_repetitions)
  data.frame(
    repetitions = repetitions,
    elapsed_seconds = elapsed,
    microseconds_per_call = 1e6 * elapsed / repetitions,
    allocations_profiled = allocations,
    allocation_repetitions = allocation_repetitions
  )
}

sampling_configs <- list(
  recursive = list(B = 1L, S = 1L, repetitions = recursive_repetitions),
  single_batch = list(B = 1L, S = 512L, repetitions = 200L),
  large_batch = list(B = 64L, S = 2048L, repetitions = 20L)
)
distributions <- list(
  symmetric_gaussian = list(nu = Inf, alpha = 0, skew_none = TRUE),
  symmetric_t = list(nu = 8, alpha = 0, skew_none = TRUE),
  skew_enabled = list(nu = 8, alpha = 0.75, skew_none = FALSE)
)

sampling_results <- list()
for (device in available_devices) {
  for (config_name in names(sampling_configs)) {
    config <- sampling_configs[[config_name]]
    for (distribution_name in names(distributions)) {
      distribution <- distributions[[distribution_name]]
      fast_output <- make_output(
        B = config$B, M = 6L, d = 2L,
        nu_value = distribution$nu,
        alpha_value = distribution$alpha,
        skew_none = distribution$skew_none,
        device = device
      )
      paths <- if (distribution$skew_none) {
        list(fast = fast_output, general_zero_alpha = within(
          fast_output, skew_none <- FALSE
        ))
      } else {
        list(general_skew = fast_output)
      }
      for (path_name in names(paths)) {
        output <- paths[[path_name]]
        result <- benchmark_call(
          function() {
            MST.PMDN::sample_mst_pmdn(
              output, num_samples = config$S, device = device
            )
          },
          repetitions = config$repetitions,
          device = device
        )
        result$device <- device
        result$operation <- "sample"
        result$config <- config_name
        result$distribution <- distribution_name
        result$path <- path_name
        sampling_results[[length(sampling_results) + 1L]] <- result
      }
    }
  }
}
sampling_results <- do.call(rbind, sampling_results)

make_loss_graph <- function(template, target_values, train_nu, train_alpha) {
  clone_leaf <- function(x, requires_grad) {
    x <- x$detach()$clone()
    if (requires_grad) {
      x$requires_grad_(TRUE)
    }
    x
  }
  output <- list(
    pi = clone_leaf(template$pi, TRUE),
    mu = clone_leaf(template$mu, TRUE),
    scale_chol = clone_leaf(template$scale_chol, TRUE),
    nu = clone_leaf(template$nu, train_nu),
    alpha = clone_leaf(template$alpha, train_alpha),
    skew_none = template$skew_none
  )
  list(output = output, target = target_values)
}

loss_results <- list()
for (device in available_devices) {
  target <- torch::torch_zeros(
    c(256, 2), dtype = torch::torch_float(), device = device
  )
  for (distribution_name in names(distributions)) {
    distribution <- distributions[[distribution_name]]
    template <- make_output(
      B = 256L, M = 6L, d = 2L,
      nu_value = distribution$nu,
      alpha_value = distribution$alpha,
      skew_none = distribution$skew_none,
      device = device
    )
    paths <- if (distribution$skew_none) {
      list(fast = template, general_zero_alpha = within(
        template, skew_none <- FALSE
      ))
    } else {
      list(general_skew = template)
    }
    for (path_name in names(paths)) {
      path_template <- paths[[path_name]]
      forward_result <- benchmark_call(
        function() {
          graph <- make_loss_graph(
            path_template, target,
            train_nu = is.finite(distribution$nu),
            train_alpha = !distribution$skew_none
          )
          MST.PMDN::loss_mst_pmdn(graph$output, graph$target)
        },
        repetitions = 500L,
        device = device
      )
      forward_result$device <- device
      forward_result$operation <- "loss_forward"
      forward_result$config <- "B256_M6_d2"
      forward_result$distribution <- distribution_name
      forward_result$path <- path_name
      loss_results[[length(loss_results) + 1L]] <- forward_result

      backward_result <- benchmark_call(
        function() {
          graph <- make_loss_graph(
            path_template, target,
            train_nu = is.finite(distribution$nu),
            train_alpha = !distribution$skew_none
          )
          loss <- MST.PMDN::loss_mst_pmdn(graph$output, graph$target)
          loss$backward()
        },
        repetitions = 100L,
        device = device
      )
      backward_result$device <- device
      backward_result$operation <- "loss_forward_backward"
      backward_result$config <- "B256_M6_d2"
      backward_result$distribution <- distribution_name
      backward_result$path <- path_name
      loss_results[[length(loss_results) + 1L]] <- backward_result
    }
  }
}
loss_results <- do.call(rbind, loss_results)

results <- rbind(sampling_results, loss_results)
results$relative_speedup <- NA_real_
group_columns <- c("device", "operation", "config", "distribution")
groups <- interaction(results[group_columns], drop = TRUE)
for (group in unique(groups)) {
  rows <- which(groups == group)
  fast <- rows[results$path[rows] == "fast"]
  general <- rows[results$path[rows] == "general_zero_alpha"]
  if (length(fast) == 1L && length(general) == 1L) {
    results$relative_speedup[fast] <-
      results$elapsed_seconds[general] / results$elapsed_seconds[fast]
  }
}

results <- results[
  c(
    "device", "operation", "config", "distribution", "path",
    "repetitions", "elapsed_seconds", "microseconds_per_call",
    "allocations_profiled", "allocation_repetitions", "relative_speedup"
  )
]
output_path <- Sys.getenv("MST_PMDN_BENCH_OUTPUT", "")
if (nzchar(output_path)) {
  utils::write.csv(results, output_path, row.names = FALSE)
}
print(results, row.names = FALSE)
