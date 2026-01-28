\name{quantile_marginal_mst_pmdn}
\alias{quantile_marginal_mst_pmdn}
\title{Estimate marginal quantiles from an MST-PMDN predictive mixture}
\usage{
quantile_marginal_mst_pmdn(
  mdn_output,
  probs,
  var_index = NULL,
  num_samples = 1000,
  device = "cpu",
  seed = NULL,
  draws = NULL
)
}
\arguments{
  \item{mdn_output}{A \code{list} as returned by \code{\link{predict_mst_pmdn}} or a forward pass of an \code{mst_pmdn_model}, containing at minimum the following named \code{\link[torch]{torch_tensor}} elements: \code{pi}, \code{mu}, \code{scale_chol}, \code{nu}, and \code{alpha}.}
  \item{probs}{Numeric matrix or \pkg{torch} tensor with shape \eqn{B \times V} containing probabilities in \eqn{[0, 1]} for each batch item and output dimension.}
  \item{var_index}{Integer vector of output dimension indices to summarize. Defaults to all output dimensions.}
  \item{num_samples}{Integer; number of Monte Carlo draws per input case. Defaults to \code{1000}.}
  \item{device}{Character; a \pkg{torch} device on which to perform sampling (e.g. \code{"cpu"} or \code{"cuda"}). Defaults to \code{"cpu"}.}
  \item{seed}{Optional integer; random seed for reproducible Monte Carlo draws.}
  \item{draws}{Optional precomputed draws as a \pkg{torch} tensor, array, or list with a \code{samples} entry (e.g. the output of \code{\link{sample_mst_pmdn}}). When provided, sampling is skipped.}
}
\description{
Draws Monte Carlo samples from the predictive MST-PMDN mixture distribution and
computes marginal quantiles for each requested output dimension. This is a
sampling-based approximation to the mixture distribution; note that
the quantile of a mixture distribution is not, in general, the mixture of
component-wise quantiles. This function therefore uses Monte Carlo sampling to
estimate the desired marginal quantiles.

For comparable results across chained sampling-based summaries, either pass the same
\code{seed} to each call or reuse identical \code{draws}.
}
\value{
A numeric matrix of shape \eqn{B \times V} (batch size by output dimension),
aligned with the columns of \code{probs} and \code{var_index}.
}
\seealso{
\code{\link{cdf_marginal_mst_pmdn}}, \code{\link{sample_mst_pmdn}}
}
