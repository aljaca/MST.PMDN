################################################################################
# Dependency-free base graphics for MST-PMDN explanation objects              #
################################################################################

plot.mst_pmdn_functional <- function(x, ...) {
  graphics::plot(
    x$data$row,
    x$data$value,
    type = "l",
    xlab = "Prediction row",
    ylab = x$functional$type,
    ...
  )
  invisible(x)
}

plot.mst_pmdn_ale <- function(x,
                              type = c("total", "channels"),
                              ...) {
  type <- match.arg(type)
  data <- x$data
  if (type == "total" || !length(x$active_channels)) {
    graphics::plot(
      data$feature_value,
      data$ale,
      type = "b",
      xlab = x$feature$name,
      ylab = paste("ALE of", x$functional$type),
      ...
    )
    graphics::abline(h = 0, lty = 2, col = "grey60")
    return(invisible(x))
  }

  old_par <- graphics::par(no.readonly = TRUE)
  on.exit(graphics::par(old_par), add = TRUE)
  panels <- c("ale", paste0("ale_", x$active_channels))
  labels <- c("total", x$active_channels)
  graphics::par(
    mfrow = c(length(panels), 1L),
    mar = c(2.5, 4, 1.5, 1),
    oma = c(0, 0, 2, 0)
  )
  for (i in seq_along(panels)) {
    graphics::plot(
      data$feature_value,
      data[[panels[i]]],
      type = "b",
      xlab = if (i == length(panels)) x$feature$name else "",
      ylab = labels[i],
      ...
    )
    graphics::abline(h = 0, lty = 2, col = "grey60")
  }
  graphics::mtext(
    sprintf(
      "Maximum |sum-to-total residual|: %s",
      format(x$diagnostics$max_abs_sum_to_total_residual, digits = 4)
    ),
    side = 3,
    outer = TRUE
  )
  invisible(x)
}

plot.mst_pmdn_ice <- function(x,
                              type = c("ice", "plate"),
                              ...) {
  type <- match.arg(type)
  if (type == "plate") {
    if (all(is.na(x$plate$local_slope))) {
      stop("Plate-style slopes require derivative = TRUE in ice_mst_pmdn().",
           call. = FALSE)
    }
    graphics::plot(
      x$plate$baseline_contrast,
      x$plate$local_slope,
      pch = 19,
      xlab = "Baseline contrast",
      ylab = "Local slope",
      ...
    )
    graphics::abline(h = 0, v = 0, lty = 2, col = "grey70")
    return(invisible(x))
  }

  y_range <- range(x$curves$centred, finite = TRUE)
  graphics::plot(
    range(x$grid),
    y_range,
    type = "n",
    xlab = x$feature$name,
    ylab = paste("Centred ICE of", x$functional$type),
    ...
  )
  for (case in x$cases) {
    block <- x$curves[x$curves$case == case, ]
    graphics::lines(
      block$feature_value, block$centred,
      col = grDevices::adjustcolor("grey35", alpha.f = 0.25)
    )
  }
  ale <- x$ale$data
  reference_ale <- stats::approx(
    ale$feature_value, ale$ale, xout = x$reference, rule = 2
  )$y
  graphics::lines(
    ale$feature_value,
    ale$ale - reference_ale,
    col = "black",
    lwd = 3
  )
  graphics::abline(h = 0, lty = 2, col = "grey60")
  invisible(x)
}

plot.mst_pmdn_decomposition <- function(x, row = 1L, ...) {
  if (!is.numeric(row) || length(row) != 1L || row < 1L ||
      row > nrow(x$data) || row != floor(row)) {
    stop("row must select one decomposition row.", call. = FALSE)
  }
  if (!length(x$active_channels)) {
    graphics::plot.new()
    graphics::title("No active parameter channels")
    return(invisible(x))
  }
  values <- vapply(
    x$active_channels,
    function(channel) x$data[[paste0("channel_", channel)]][row],
    numeric(1)
  )
  graphics::barplot(
    values,
    names.arg = x$active_channels,
    ylab = paste("Contribution to", x$functional$type),
    main = sprintf(
      "Total %s; residual %s",
      format(x$data$total[row], digits = 4),
      format(x$data$sum_to_total_residual[row], digits = 3)
    ),
    ...
  )
  graphics::abline(h = 0, col = "grey40")
  invisible(x)
}

plot.mst_pmdn_image_contrast <- function(x,
                                         type = c("distribution", "cases", "channels"),
                                         row = 1L,
                                         ...) {
  type <- match.arg(type)
  if (type == "distribution") {
    graphics::hist(
      x$data$contrast,
      xlab = paste("Whole-image contrast in", x$functional$type),
      main = "",
      ...
    )
    graphics::abline(v = 0, lty = 2, col = "grey50")
  } else if (type == "cases") {
    graphics::plot(
      x$data$case,
      x$data$contrast,
      type = "h",
      xlab = "Case",
      ylab = "Whole-image contrast",
      ...
    )
    graphics::abline(h = 0, lty = 2, col = "grey50")
  } else {
    if (!length(x$active_channels)) {
      stop("Channel plotting requires decompose = TRUE and active channels.",
           call. = FALSE)
    }
    if (!is.numeric(row) || length(row) != 1L || !is.finite(row) ||
        row < 1L || row > nrow(x$data) || row != floor(row)) {
      stop("row must select one image-contrast row.", call. = FALSE)
    }
    values <- vapply(
      x$active_channels,
      function(channel) x$data[[paste0("channel_", channel)]][row],
      numeric(1)
    )
    graphics::barplot(
      values,
      names.arg = x$active_channels,
      ylab = "Whole-image contribution",
      ...
    )
    graphics::abline(h = 0, col = "grey50")
    graphics::mtext(
      sprintf(
        "Sum-to-total residual: %s",
        format(x$data$sum_to_total_residual[row], digits = 4)
      ),
      side = 3,
      line = 0.25
    )
  }
  invisible(x)
}

.occlusion_map_matrix_mst_pmdn <- function(data, value_column) {
  rows <- sort(unique(data$row_center))
  columns <- sort(unique(data$col_center))
  z <- matrix(NA_real_, nrow = length(columns), ncol = length(rows))
  for (i in seq_len(nrow(data))) {
    z[
      match(data$col_center[i], columns),
      match(data$row_center[i], rows)
    ] <- data[[value_column]][i]
  }
  list(rows = rows, columns = columns, z = z)
}

plot.mst_pmdn_image_occlusion <- function(x,
                                          case = NULL,
                                          group = NULL,
                                          statistic = NULL,
                                          ...) {
  if (is.null(group)) group <- names(x$channel_groups)[1L]
  if (!group %in% names(x$channel_groups)) {
    stop("group is not present in the occlusion object.", call. = FALSE)
  }
  if (is.null(case)) {
    data <- x$population[x$population$group == group, , drop = FALSE]
    if (is.null(statistic)) statistic <- "mean_signed_effect"
  } else {
    data <- x$data[
      x$data$group == group & x$data$case == case,
      ,
      drop = FALSE
    ]
    if (!nrow(data)) stop("case is not present in the occlusion object.",
                          call. = FALSE)
    if (is.null(statistic)) statistic <- "effect"
  }
  if (!is.character(statistic) || length(statistic) != 1L ||
      !statistic %in% names(data) || !is.numeric(data[[statistic]])) {
    stop("statistic must name a numeric occlusion output column.",
         call. = FALSE)
  }
  map <- .occlusion_map_matrix_mst_pmdn(data, statistic)
  limit <- max(abs(map$z), na.rm = TRUE)
  if (!is.finite(limit) || limit == 0) limit <- 1
  colours <- grDevices::colorRampPalette(
    c("#2166AC", "white", "#B2182B")
  )(101L)
  graphics::image(
    x = map$columns,
    y = map$rows,
    z = map$z,
    col = colours,
    zlim = c(-limit, limit),
    xlab = "Image column",
    ylab = "Image row",
    useRaster = TRUE,
    ...
  )
  if (!is.null(case) && startsWith(statistic, "channel_")) {
    graphics::mtext(
      sprintf(
        "Maximum |sum-to-total residual|: %s",
        format(
          .max_abs_finite_mst_pmdn(data$sum_to_total_residual),
          digits = 4
        )
      ),
      side = 3,
      line = 0.25
    )
  }
  invisible(x)
}

plot.mst_pmdn_tail_components <- function(x, row = 1L, ...) {
  data <- x$data[x$data$row == row, , drop = FALSE]
  if (!nrow(data)) stop("row is not present in the tail-component object.",
                        call. = FALSE)
  old_par <- graphics::par(no.readonly = TRUE)
  on.exit(graphics::par(old_par), add = TRUE)
  graphics::par(mfrow = c(1L, 3L), mar = c(4, 4, 2, 1))
  labels <- data$component
  graphics::barplot(data$weight, names.arg = labels, main = "Weight",
                    ylab = expression(pi[g]), ...)
  graphics::barplot(
    data$component_probability,
    names.arg = labels,
    main = "Within-component",
    ylab = expression(p[g]),
    ...
  )
  graphics::barplot(
    data$contribution,
    names.arg = labels,
    main = "Contribution",
    ylab = expression(pi[g] * p[g]),
    ...
  )
  invisible(x)
}
