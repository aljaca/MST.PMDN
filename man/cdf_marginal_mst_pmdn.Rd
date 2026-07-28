\name{cdf_marginal_mst_pmdn}
\alias{cdf_marginal_mst_pmdn}
\title{Monte Carlo estimation of marginal CDFs for MST-PMDN mixtures}
\usage{
cdf_marginal_mst_pmdn(
  mdn_output,
  y,
  var_index = NULL,
  num_samples = 1000,
  device = "cpu",
  seed = NULL,
  draws = NULL
)
}
\arguments{
  \item{mdn_output}{A \code{list} as returned by \code{\link{predict_mst_pmdn}} containing mixture parameters. The optional \code{skew_none} field is passed through when Monte Carlo draws are generated; if absent, sampling uses the general skew path. In a hand-built list, \code{skew_none = TRUE} is valid only when \code{alpha} is structurally zero.}
  \item{y}{Numeric vector of response values at which to evaluate the marginal CDF.}
  \item{var_index}{Integer; indices of the response dimensions to evaluate. Defaults to all output dimensions.}
  \item{num_samples}{Integer; number of Monte Carlo draws used to approximate the marginal CDF.}
  \item{device}{Character; a \pkg{torch} device on which to perform sampling (e.g. \code{"cpu"} or \code{"cuda"}).}
  \item{seed}{Optional integer; random seed for reproducible Monte Carlo draws.}
  \item{draws}{Optional precomputed draws as a \pkg{torch} tensor, array, or list with a \code{samples} entry (e.g. the output of \code{\link{sample_mst_pmdn}}). When provided, sampling is skipped.}
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

For comparable results across chained sampling-based summaries, either pass the same
\code{seed} to each call or reuse identical \code{draws}.
}
\value{
A \code{torch_tensor} of estimated marginal CDF values with rows aligned to the prediction batch and columns aligned to the selected output dimensions.
}
\seealso{
\code{\link{quantile_marginal_mst_pmdn}}, \code{\link{sample_mst_pmdn}}
}
