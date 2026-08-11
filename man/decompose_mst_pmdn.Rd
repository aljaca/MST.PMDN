\name{decompose_mst_pmdn}
\alias{decompose_mst_pmdn}
\title{Shapley-decompose a one-component MST-PMDN functional contrast}
\description{
Allocates a scalar functional contrast between two one-component predictive
states among complete location, scale, skewness, and degrees-of-freedom blocks.
}
\usage{
decompose_mst_pmdn(
  pred_from,
  pred_to,
  functional,
  channels = c("location", "scale", "skewness", "df"),
  method = "shapley",
  latent_draws = NULL,
  num_samples = 4096L,
  seed = NULL,
  chunk_size = NULL,
  device = "cpu",
  response_names = NULL,
  min_tail_draws = 20L
)
}
\arguments{
  \item{pred_from, pred_to}{Matching one-component prediction lists.}
  \item{functional}{Object returned by \code{\link{mst_functional}}.}
  \item{channels}{Requested subset of \code{"location"}, \code{"scale"},
  \code{"skewness"}, and \code{"df"}.}
  \item{method}{Currently \code{"shapley"}.}
  \item{latent_draws}{Optional common latent bank.}
  \item{num_samples}{Monte Carlo draw count.}
  \item{seed}{Optional latent-bank seed.}
  \item{chunk_size}{Optional case chunk size.}
  \item{device}{Torch evaluation device.}
  \item{response_names}{Optional prediction response names.}
  \item{min_tail_draws}{Positive warning threshold for expected tail draws.}
}
\details{
Every hybrid prediction takes each selected parameter block from
\code{pred_to} and every other block from \code{pred_from}. With all four
channels active, exact Shapley averaging uses 16 hybrid functional evaluations.
The complete Cholesky scale factor is one block. Because skew direction is
defined in standardized coordinates, changing that factor can rotate the skew
direction in response space; the scale channel therefore includes that induced
directional change as well as marginal scale and dependence. This is
parameter-channel attribution, not feature SHAP.

Unchanged channels, exact fixed degrees of freedom, and effective zero
skewness disappear automatically; structural symmetry and an exactly zero
numerical skew vector are treated identically. Diagnostics report maximum
absolute endpoint changes in location, Cholesky scale, skewness, and inverse
degrees of freedom. Hybrid states are evaluated within row chunks so repeated
finite-df parents reuse their Gamma-quantile transforms. Full decomposition is rejected for mixtures
because head labels and compensating component changes are not identified.
}
\value{
An \code{mst_pmdn_decomposition} object containing a data frame of endpoints,
total contrast, active-channel contributions, sum-to-total residual, parameter
change magnitudes, and aggregated Monte Carlo diagnostics.
}
