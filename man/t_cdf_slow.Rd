\name{t_cdf_slow}
\alias{t_cdf_slow}
\title{Accurate but slower Student t CDF approximation}
\docType{function}
\usage{
t_cdf_slow(z, nu)
}
\arguments{
  \item{z}{Numeric vector or \code{\link[torch]{torch_tensor}} of quantiles.}
  \item{nu}{Degrees of freedom (scalar or tensor).}
}
\description{
Computes a more accurate but computationally slower approximation to the Student t
CDF by scaling and using the error function under the hood. Useful when precision is
more important than speed.
}
\value{
A \code{\link[torch]{torch_tensor}} of CDF values corresponding to \code{z}.
}
\examples{
library(torch)
z <- torch_tensor(c(-2, -0.5, 0.5, 2))
nu <- 10
t_cdf_slow(z, nu)
}
