\name{loss_mst_pmdn}
\alias{loss_mst_pmdn}
\title{Negative Log-Likelihood Loss for MST-PMDN}
\usage{
loss_mst_pmdn(output, target, lambda_alpha = 0, lambda_nu_inv = 0)
}
\arguments{
  \item{output}{A list as returned by a forward pass of an \code{mst_pmdn_model}, containing at least the following named \code{\link[torch]{torch_tensor}} elements:
    \code{pi}, \code{mu}, \code{scale_chol}, \code{nu}, and \code{alpha}.}
  \item{target}{A \code{\link[torch]{torch_tensor}} or numeric matrix of true responses, with shape \code{[batch_size, d]}.}
  \item{lambda_alpha}{Numeric scalar; L2 penalty weight applied to the final skewness parameters \code{alpha}.}
  \item{lambda_nu_inv}{Numeric scalar; penalty weight applied to \code{(1/nu)^2}.}
}
\description{
Computes the average negative log-likelihood under a mixture of multivariate skew t distributions for a batch of examples.
The function extracts mixture weights, component locations, Cholesky scale factors, degrees of freedom, and skewness parameters from \code{output}, then evaluates the MST-PMDN log-density at \code{target}, finally averaging (and negating) to produce the loss.
}
\value{
A single-element \code{\link[torch]{torch_tensor}} containing the batch-averaged negative log-likelihood.
}
