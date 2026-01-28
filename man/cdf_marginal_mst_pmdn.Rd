\name{cdf_marginal_mst_pmdn}
\alias{cdf_marginal_mst_pmdn}
\title{Monte Carlo estimation of marginal CDFs for MST-PMDN mixtures}
\usage{
cdf_marginal_mst_pmdn(mdn_output, y, var_index = NULL, num_samples = 1000, device = "cpu")
}
\arguments{
  \item{mdn_output}{A \code{list} as returned by \code{\link{predict_mst_pmdn}} containing mixture parameters.}
  \item{y}{Numeric vector of response values at which to evaluate the marginal CDF.}
  \item{var_index}{Integer; indices of the response dimensions to evaluate. Defaults to all output dimensions.}
  \item{num_samples}{Integer; number of Monte Carlo draws used to approximate the marginal CDF.}
  \item{device}{Character; a \pkg{torch} device on which to perform sampling (e.g. \code{"cpu"} or \code{"cuda"}).}
}
\description{
Approximates the marginal CDF for a selected response dimension using Monte Carlo draws from
an MST-PMDN predictive mixture.
}
\details{
The mixture marginal CDF is
\deqn{F(y_j) = \sum_{k=1}^G \pi_k F_{k,j}(y_j),}
where \eqn{F_{k,j}} is the univariate skew-t marginal CDF for component \eqn{k} and
response dimension \eqn{j}. The component skew-t marginals do not have closed-form CDFs in
general, so Monte Carlo estimation is recommended. The helper draws from the mixture (via
\code{\link{sample_mst_pmdn}}) and estimates \eqn{F(y_j)} by the empirical proportion of samples
below \eqn{y_j}.
}
\value{
A numeric vector of estimated marginal CDF values, aligned with \code{y}.
}
\seealso{
\code{\link{quantile_marginal_mst_pmdn}}, \code{\link{sample_mst_pmdn}}
}
