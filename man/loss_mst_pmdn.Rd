\name{loss_mst_pmdn}
\alias{loss_mst_pmdn}
\title{Negative Log-Likelihood Loss for MST-PMDN}
\usage{
loss_mst_pmdn(output, target, lambda_alpha = 0, lambda_nu_inv = 0)
}
\arguments{
  \item{output}{A list as returned by a forward pass of an \code{mst_pmdn_model}, containing at least the following named \code{\link[torch]{torch_tensor}} elements:
    \code{pi}, \code{mu}, \code{scale_chol}, \code{nu}, and \code{alpha}.
    The optional single logical field \code{skew_none = TRUE} selects the
    symmetric fast path. If the field is absent, the general skew factor is
    evaluated. For a hand-built list, setting \code{skew_none = TRUE} requires
    \code{alpha} to be structurally zero; the caller is responsible for
    maintaining this invariant.}
  \item{target}{A \code{\link[torch]{torch_tensor}} or numeric matrix of true responses, with shape \code{[batch_size, d]}.}
  \item{lambda_alpha}{Single finite non-negative numeric value; L2 penalty weight applied to the final skewness parameters \code{alpha}.}
  \item{lambda_nu_inv}{Single finite non-negative numeric value; penalty weight applied to \code{(1/nu)^2}. Exact normal components contribute zero.}
}
\description{
Computes the average negative log-likelihood under a mixture of multivariate skew t and exact Gaussian/skew-normal components for a batch of examples.
The function extracts mixture weights, component locations, Cholesky scale factors, degrees of freedom, and skewness parameters from \code{output}, then evaluates the MST-PMDN log-density at \code{target}, finally averaging (and negating) to produce the loss. Finite-\code{nu} components use a cancellation-resistant multivariate t normalizing constant and Hill's Student t approximation through a direct, stable log-CDF. Components with \code{nu = Inf} use the exact Gaussian kernel and normal log-CDF. Outputs with \code{skew_none = TRUE} bypass the skew-factor calculation. Model-generated outputs satisfy the required invariant that \code{alpha} is structurally zero; hand-built outputs must satisfy it explicitly.
}
\value{
A single-element \code{\link[torch]{torch_tensor}} containing the batch-averaged negative log-likelihood.
}
