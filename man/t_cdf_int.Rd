\name{t_cdf_int}
\alias{t_cdf_int}
\title{Numerical integration–based Student t CDF}
\description{
Computes the Student t cumulative distribution function by numerically integrating the probability density function.
}
\usage{
t_cdf_int(t_val, nu, num_integration_points = 1000L)
}
\arguments{
  \item{\code{t_val}}{Numeric vector or \code{\link[torch]{torch_tensor}} of quantiles at which to evaluate the CDF.}
  \item{\code{nu}}{Degrees of freedom (scalar or tensor).}
  \item{\code{num_integration_points}}{Integer; number of points to use in the numerical integration. Defaults to \code{1000L}.}
}
\value{
A \code{\link[torch]{torch_tensor}} of CDF values corresponding to \code{t_val}.
}
