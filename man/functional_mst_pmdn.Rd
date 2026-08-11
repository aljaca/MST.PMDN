\name{functional_mst_pmdn}
\alias{functional_mst_pmdn}
\title{Evaluate a scalar MST-PMDN predictive functional}
\usage{
functional_mst_pmdn(
  pred,
  functional,
  num_samples = 4096L,
  latent_draws = NULL,
  seed = NULL,
  chunk_size = NULL,
  device = "cpu",
  response_names = NULL,
  min_tail_draws = 20L
)
}
\arguments{
  \item{pred}{Prediction list returned by \code{\link{predict_mst_pmdn}}.}
  \item{functional}{Specification returned by \code{\link{mst_functional}}.}
  \item{num_samples}{Positive integer Monte Carlo sample count for tail
  functionals.}
  \item{latent_draws}{Optional bank from
  \code{\link{latent_draws_mst_pmdn}}. Its draw count takes precedence over
  \code{num_samples}.}
  \item{seed}{Optional seed used when a bank is created internally.}
  \item{chunk_size}{Optional positive number of prediction rows evaluated at
  once. \code{NULL} chooses a memory-bounded adaptive value that accounts for
  the quadratic scale-factor gather in the response dimension.}
  \item{device}{Torch evaluation device.}
  \item{response_names}{Optional names for prediction response dimensions.
  Used when \code{functional$responses} is character.}
  \item{min_tail_draws}{Warning threshold for the expected number of draws in
  the relevant tail. Warnings have class
  \code{mst_pmdn_tail_resolution_warning}.}
}
\description{
Evaluates one scalar per prediction row. Means, variances, standard deviations,
covariances, and correlations use exact mixture moments. Quantiles, marginal
and joint exceedances, tail spread, and tail asymmetry use stable common-random-
number Monte Carlo evaluation.
}
\value{
An object of class \code{mst_pmdn_functional}. Its \code{data} member is a
plain data frame with the row, value, expected tail draws, and a low-resolution
flag. Settings, diagnostics, and the latent bank are retained separately.
}
\details{
Exact moments include skewness, finite degrees of freedom, and between-component
covariance. Means are undefined for finite \eqn{\nu\leq 1}; covariance-based
functionals are undefined for finite \eqn{\nu\leq 2}. Exact Gaussian and
skew-normal components use \eqn{\nu=\infty}.

Tail asymmetry is normalized by its selected interquantile spread and is thus
dimensionless. Common component uniforms deliberately correlate Monte Carlo
error across prediction rows. Expected tail draws diagnose only the complete
marginal functional; expected component draw counts are retained separately for
mixtures. Higher-level ALE, ICE, decomposition, contrast, and occlusion calls
muffle only this class internally and emit one aggregated warning while
preserving unrelated warnings.
}
