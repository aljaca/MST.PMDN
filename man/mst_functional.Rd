\name{mst_functional}
\alias{mst_functional}
\title{Define a scalar MST-PMDN predictive functional}
\description{
Defines a validated scalar scientific summary of an MST-PMDN predictive
distribution.
}
\usage{
mst_functional(
  type,
  responses = NULL,
  prob = NULL,
  threshold = NULL,
  direction = "upper",
  transform = "identity"
)
}
\arguments{
  \item{type}{One of \code{"mean"}, \code{"variance"}, \code{"sd"},
  \code{"covariance"}, \code{"correlation"}, \code{"quantile"},
  \code{"exceedance"}, \code{"joint_exceedance"}, \code{"tail_spread"},
  \code{"tail_asymmetry"}, \code{"df"}, \code{"inverse_df"}, or
  \code{"skew_direction"}.}
  \item{responses}{Response names or R 1-based response indices. Marginal
  functionals require one, pairwise functionals require two, and degrees-of-
  freedom diagnostics require none.}
  \item{prob}{One quantile probability, or increasing lower and upper
  probabilities for tail spread or asymmetry.}
  \item{threshold}{One threshold for a marginal exceedance or two thresholds,
  ordered as \code{responses}, for a joint exceedance.}
  \item{direction}{\code{"upper"} or \code{"lower"} for each event.}
  \item{transform}{\code{"identity"}, \code{"log"}, \code{"logit"} for
  event probabilities, or \code{"fisher_z"} for correlation.}
}
\details{
Means and covariance-based summaries are analytic. Component skew-t moments
are combined with mixture weights and between-component covariance. Means are
undefined for finite \eqn{\nu \le 1}; variances and covariances are undefined
for finite \eqn{\nu \le 2}. The exact \eqn{\nu=\infty} skew-normal or Gaussian
limit is handled directly.

Quantiles, exceedances, tail spread, and tail asymmetry use stable Monte Carlo
evaluation. Returned diagnostics include the expected number of draws in the
relevant tail; a nominal 0.999 quantile with 4096 draws has only 4.096 expected
upper-tail draws.
}
\value{
An object of class \code{mst_functional}.
}
