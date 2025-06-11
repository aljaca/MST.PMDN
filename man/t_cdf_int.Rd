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
}
\value{
A \code{\link[torch]{torch_tensor}} containing the integrated CDF values.
}
