\name{t_cdf_slow}
\alias{t_cdf_slow}
\title{Accurate but slower Student t CDF approximation}
\usage{
t_cdf_slow(z, nu)
}
\arguments{
  \item{z}{Numeric vector or \code{\link[torch]{torch_tensor}} of quantiles.}
  \item{nu}{Degrees of freedom (scalar or tensor).}
}
\description{
Computes a more accurate but computationally slower approximation to the Student t
CDF using \code{pt} and finite differences for the gradient.
}
\value{
A \code{\link[torch]{torch_tensor}} of CDF values corresponding to \code{z}.
}
