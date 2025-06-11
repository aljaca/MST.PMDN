\name{t_cdf_fast}
\alias{t_cdf_fast}
\title{Fast approximation of the Student t CDF}
\docType{function}
\usage{
t_cdf_fast(z, nu)
}
\arguments{
  \item{z}{Numeric vector or \code{\link[torch]{torch_tensor}} of quantiles.}
  \item{nu}{Degrees of freedom (scalar or tensor).}
}
\description{
Computes a differentiable “fast” approximation to the cumulative distribution function
of the Student t distribution, suitable for use in \pkg{torch} computation graphs.
}
\value{
A \code{\link[torch]{torch_tensor}} of the same shape as \code{z}, containing CDF values.
}
\examples{
library(torch)
z <- torch_tensor(c(-1, 0, 1))
nu <- 5
t_cdf_fast(z, nu)
}
