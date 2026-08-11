\name{ale_mst_pmdn}
\alias{ale_mst_pmdn}
\title{Accumulated local effects for an MST-PMDN functional}
\description{
Computes one-dimensional accumulated local effects (ALE) for a scalar
predictive functional.
}
\usage{
ale_mst_pmdn(
  model,
  inputs,
  feature,
  functional,
  image_inputs = NULL,
  n_bins = 20L,
  num_samples = 4096L,
  decompose = FALSE,
  channels = c("location", "scale", "skewness", "df"),
  latent_draws = NULL,
  seed = NULL,
  chunk_size = NULL,
  device = "cpu",
  response_names = NULL,
  min_tail_draws = 20L
)
}
\arguments{
  \item{model}{A fitted MST-PMDN torch module.}
  \item{inputs}{Numeric input matrix, data frame, or two-dimensional tensor.}
  \item{feature}{A named input column or R 1-based column index.}
  \item{functional}{Object returned by \code{\link{mst_functional}}.}
  \item{image_inputs}{Optional case-matched image tensor or array. Each image
  is held fixed when its tabular feature is perturbed.}
  \item{n_bins}{Requested number of empirical ALE bins. Tied quantiles can
  reduce the actual number.}
  \item{num_samples}{Monte Carlo draws for sampled functionals.}
  \item{decompose}{If \code{TRUE}, decompose each local ALE contrast among
  one-component parameter channels.}
  \item{channels}{Requested parameter channels.}
  \item{latent_draws}{Optional common latent bank.}
  \item{seed}{Optional latent-bank seed.}
  \item{chunk_size}{Optional prediction and functional case chunk size.}
  \item{device}{Torch evaluation device.}
  \item{response_names}{Optional prediction response names.}
  \item{min_tail_draws}{Positive warning threshold for expected tail draws.}
}
\details{
ALE replaces a feature by its lower and upper empirical-bin boundaries only
for rows observed in that bin, merging empirically empty bins when necessary.
Local differences are averaged, accumulated at bin midpoints, and empirically
centred. The midpoint value subtracts half of the current bin effect, a
within-bin linear interpolation convention rather than the boundary-valued
definition in Apley and Zhu. This makes ALE the preferred population summary
when meteorological predictors are correlated. For Monte Carlo functionals,
diagnostics report the minimum expected tail-draw count actually encountered at
the evaluated bin endpoints or hybrid parameter states. Internal warnings are
aggregated into one classed tail-resolution warning per ALE call when the
minimum falls below \code{min_tail_draws}.
}
\value{
An \code{mst_pmdn_ale} object with \code{data}, breaks, settings, and
diagnostics.
}
