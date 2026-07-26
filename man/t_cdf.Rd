\name{t_cdf}
\alias{t_cdf}
\title{Differentiable Student t CDF Approximation}
\usage{
t_cdf(z, nu)
}
\arguments{
  \item{z}{Numeric vector or \code{\link[torch]{torch_tensor}} of quantiles.}
  \item{nu}{Degrees of freedom (scalar or tensor).}
}
\description{
Computes the Student t cumulative distribution using the Gaver--Kafadar
transformed-normal approximation for \code{nu >= 3}, with exact closed forms
for \code{nu = 1} (Cauchy) and \code{nu = 2}. The implementation is compatible
with torch computation graphs.
}
\details{
For \code{nu >= 3}, the normal quantile argument is
\deqn{
\mathop{\rm sign}(z)
\sqrt{\frac{\log(1 + z^2 / \nu)}{g(\nu)}},
\qquad
g(\nu) = \frac{\nu - 3/2}{(\nu - 1)^2}.
}
The implementation evaluates the equivalent expression without
\code{sign(z)} and uses a series expansion at \code{z = 0}. The likelihood
uses a direct log-CDF calculation with a lower-tail asymptotic expansion,
avoiding cancellation and a hard probability floor.
}
\value{
A \code{\link[torch]{torch_tensor}} of CDF values.
}
\references{
Gaver, D. P. and Kafadar, K. (1984).
A retrievable recipe for inverse t.
\emph{The American Statistician}, \strong{38}, 308--311.

Gleason, J. R. (2000).
A note on a proposed Student t approximation.
\emph{Computational Statistics & Data Analysis}, \strong{34}, 63--66.
}
