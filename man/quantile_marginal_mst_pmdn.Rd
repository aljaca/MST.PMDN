\name{quantile_marginal_mst_pmdn}
\alias{quantile_marginal_mst_pmdn}
\title{Monte Carlo estimation of marginal quantiles for MST-PMDN mixtures}
\usage{
quantile_marginal_mst_pmdn(mdn_output, probs, dim, num_samples = 1000, device = "cpu")
}
\arguments{
  \item{mdn_output}{A \code{list} as returned by \code{\link{predict_mst_pmdn}} containing mixture parameters.}
  \item{probs}{Numeric vector of probabilities in \eqn{[0, 1]} for which to estimate quantiles.}
  \item{dim}{Integer; index of the response dimension to evaluate.}
  \item{num_samples}{Integer; number of Monte Carlo draws used to approximate the marginal quantiles.}
  \item{device}{Character; a \pkg{torch} device on which to perform sampling (e.g. \code{"cpu"} or \code{"cuda"}).}
}
\description{
Approximates marginal quantiles for a selected response dimension using Monte Carlo draws from
an MST-PMDN predictive mixture.
}
\details{
Quantiles are computed by inverting the estimated mixture marginal CDF using samples drawn
from the mixture. Component quantiles do not combine linearly, so the mixture quantile is not
the weighted sum of component quantiles. Monte Carlo estimation is therefore recommended in
practice.
}
\value{
A numeric vector of estimated marginal quantiles aligned with \code{probs}.
}
\seealso{
\code{\link{cdf_marginal_mst_pmdn}}, \code{\link{sample_mst_pmdn}}
}
