\name{loss_mst_pmdn}
\alias{loss_mst_pmdn}
\title{Negative Log-Likelihood Loss for MST-PMDN}

\usage{
loss_mst_pmdn(output, target, nu_switch = 20)
}

\arguments{
  \item{\code{output}}{A list as returned by a forward pass of an \code{mst_pmdn_model}, containing at least the following named \code{\link[torch]{torch_tensor}} elements:
    \code{pi}, \code{mu}, \code{scale_chol}, \code{nu}, and \code{alpha}.}
  \item{\code{target}}{A \code{\link[torch]{torch_tensor}} or numeric matrix of true responses, with shape \code{[batch_size, d]}.}
  \item{\code{nu_switch}}{Numeric scalar. Degrees-of-freedom threshold for switching between the \code{\link{t_cdf_fast}} and \code{\link{t_cdf_slow}} implementations of the Student t CDF.}
}

\description{
Computes the average negative log-likelihood under a mixture of multivariate skew t distributions for a batch of examples.  
The function extracts mixture weights, component means, Cholesky scale factors, degrees of freedom, and skewness parameters from \code{output}, then evaluates the MST-PMDN log-density at \code{target}, finally averaging (and negating) to produce the loss.  
The \code{nu_switch} argument determines at what \code{nu} value the code uses the faster (but approximate) versus slower CDF routines.
}

\value{
A single-element \code{\link[torch]{torch_tensor}} containing the batch-averaged negative log-likelihood.
}

