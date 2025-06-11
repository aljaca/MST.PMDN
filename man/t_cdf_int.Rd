\name{t_cdf_int}
\alias{t_cdf_int}
\title{Numerical integration–based Student t CDF}
\docType{function}
\usage{
t_cdf_int(z, nu)
}
\arguments{
  \item{z}{Numeric vector or \code{\link[torch]{torch_tensor}} of quantiles.}
  \item{nu}{Degrees of freedom (scalar or tensor).}
}
\description{
Computes the Student t CDF by directly numerically integrating the PDF.
This is the most accurate but also the slowest method.
}
\value{
A \code{\link[torch]{torch_tensor}} containing the integrated CDF values.
}
\examples{
library(torch)
z <- torch_tensor(seq(-3, 3, by = 1))
nu <- 4
t_cdf_int(z, nu)
}
