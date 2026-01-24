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
  seed = NULL
)
}
\arguments{
  \item{mdn_output}{A \code{list} as returned by \code{\link{predict_mst_pmdn}} or a forward pass of an \code{mst_pmdn_model}, containing at minimum the following named \code{\link[torch]{torch_tensor}} elements: \code{pi}, \code{mu}, \code{scale_chol}, \code{nu}, and \code{alpha}.}
  \item{probs}{Numeric vector of probabilities in \eqn{[0, 1]} for which to compute quantiles.}
  \item{var_index}{Integer vector of output dimension indices to summarize. Defaults to all output dimensions.}
  \item{num_samples}{Integer; number of Monte Carlo draws per input case. Defaults to \code{1000}.}
  \item{device}{Character; a \pkg{torch} device on which to perform sampling (e.g. \code{"cpu"} or \code{"cuda"}). Defaults to \code{"cpu"}.}
  \item{seed}{Optional integer; random seed for reproducible Monte Carlo draws.}
}
\description{
Draws Monte Carlo samples from the predictive MST-PMDN mixture distribution and
computes marginal quantiles for each requested output dimension. This is a
sampling-based approximation to the mixture distribution; note that
the quantile of a mixture distribution is not, in general, the mixture of
component-wise quantiles. This function therefore uses Monte Carlo sampling to
estimate the desired marginal quantiles.
}
\value{
An array of shape \code{c(batch_size, length(var_index), length(probs))} with
dimension names \code{batch}, \code{var}, and \code{prob}.
}
