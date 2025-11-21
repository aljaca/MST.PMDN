\name{t_cdf}
\alias{t_cdf}
\title{Student t CDF approximation}
\usage{
t_cdf(z, nu)
}
\arguments{
  \item{z}{Numeric vector or \code{\link[torch]{torch_tensor}} of quantiles.}
  \item{nu}{Degrees of freedom (scalar or tensor).}
}
\description{
Computes the Student t cumulative distribution using a fast Li--De Moor corrected normal approximation for
\code{nu >= 3}, with exact closed forms for \code{nu = 1} (Cauchy) and \code{nu = 2}. The implementation
is compatible with torch computation graphs.
}
\value{
A \code{\link[torch]{torch_tensor}} of CDF values.
}
