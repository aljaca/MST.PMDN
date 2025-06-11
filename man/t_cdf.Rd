\name{t_cdf}
\alias{t_cdf}
\title{Unified interface to Student t CDF}
\docType{function}
\usage{
t_cdf(z, nu, nu_switch = 20)
}
\arguments{
  \item{z}{Numeric vector or \code{\link[torch]{torch_tensor}} of quantiles.}
  \item{nu}{Degrees of freedom (scalar or tensor).}
  \item{nu_switch}{Value of \code{nu} below/above which \code{t_cdf_slow}/\code{t_cdf_fast} are used.}
}
\description{
Dispatches to \code{\link{t_cdf_fast}} or \code{\link{t_cdf_slow}}, providing a single API
for computing the Student t cumulative distribution.
}
\value{
A \code{\link[torch]{torch_tensor}} of CDF values.
}
