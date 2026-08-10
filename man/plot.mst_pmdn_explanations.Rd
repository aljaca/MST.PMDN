\name{plot.mst_pmdn_explanations}
\alias{plot.mst_pmdn_functional}
\alias{plot.mst_pmdn_ale}
\alias{plot.mst_pmdn_ice}
\alias{plot.mst_pmdn_decomposition}
\alias{plot.mst_pmdn_image_contrast}
\alias{plot.mst_pmdn_image_occlusion}
\alias{plot.mst_pmdn_tail_components}
\title{Plot MST-PMDN explanation objects}
\usage{
\method{plot}{mst_pmdn_functional}(
  x,
  type = "l",
  xlab = "Prediction row",
  ylab = x$functional$type,
  main = NULL,
  ...
)

\method{plot}{mst_pmdn_ale}(
  x,
  type = c("total", "channels"),
  xlab = x$feature$name,
  ylab = NULL,
  main = NULL,
  ...
)

\method{plot}{mst_pmdn_ice}(
  x,
  xlab = NULL,
  ylab = NULL,
  main = NULL,
  ...
)

\method{plot}{mst_pmdn_decomposition}(
  x,
  row = 1L,
  xlab = "",
  ylab = NULL,
  main = NULL,
  ...
)

\method{plot}{mst_pmdn_image_contrast}(
  x,
  type = c("distribution", "cases", "channels"),
  row = 1L,
  xlab = NULL,
  ylab = NULL,
  main = NULL,
  ...
)

\method{plot}{mst_pmdn_image_occlusion}(
  x,
  case = NULL,
  group = NULL,
  statistic = NULL,
  xlab = "Image column",
  ylab = "Image row",
  main = NULL,
  col = NULL,
  zlim = NULL,
  useRaster = FALSE,
  ...
)

\method{plot}{mst_pmdn_tail_components}(
  x,
  row = 1L,
  xlab = "",
  ylab = expression(pi[g], p[g], pi[g] * p[g]),
  main = c("Weight", "Within-component", "Contribution"),
  ...
)
}
\arguments{
  \item{x}{An MST-PMDN explanation object.}
  \item{type}{Plot type appropriate to the explanation class.}
  \item{row}{R 1-based explanation row to display.}
  \item{case}{Optional image-occlusion case. The default displays a population
  summary.}
  \item{group}{Named image channel group.}
  \item{statistic}{Numeric occlusion output column to map.}
  \item{xlab,ylab,main,col,zlim}{Standard graphical arguments. Scalar
  labels supplied to the three-panel component plot are recycled.}
  \item{useRaster}{Logical passed to \code{\link[graphics]{image}}. It defaults
  to \code{FALSE} so patch-centre grids need not be regular.}
  \item{...}{Additional graphical arguments.}
}
\description{
Base-graphics methods for predictive functionals, tabular effects, image
effects, parameter-channel decompositions, and mixture tail components.
}
\value{
The input object, invisibly.
}
