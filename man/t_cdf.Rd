\name{t_cdf}
\alias{t_cdf}
\title{Differentiable Student t CDF Approximation}
\usage{
t_cdf(z, nu)
}
\arguments{
  \item{z}{Numeric vector or \code{\link[torch]{torch_tensor}} of quantiles.}
  \item{nu}{Degrees of freedom (scalar or tensor); \code{Inf} selects the exact normal limit.}
}
\description{
Computes the Student t cumulative distribution using Hill's transformed-normal
approximation for \code{nu >= 3}, with exact closed forms for \code{nu = 1}
(Cauchy) and \code{nu = 2}, and the exact normal CDF for \code{nu = Inf}.
The implementation is compatible with torch
computation graphs.
}
\details{
For \code{nu >= 3}, define
\deqn{
r = (\nu - 1/2)\log(1 + z^2/\nu)
\quad\hbox{and}\quad
b = 48(\nu - 1/2)^2.
}
Writing \eqn{w = \mathrm{sign}(z)\sqrt{r}}, Hill's normal quantile
argument is
\deqn{
w + \frac{w^3 + 3w}{b}
- \frac{4w^7 + 33w^5 + 240w^3 + 855w}
{10b(b + 0.8w^4 + 100)}.
}
The implementation uses the algebraic form reported by Brophy (1987), factors
the signed leading term through \code{z}, and uses a series for
\code{log1p(u)/u} at \code{u = 0}. This removes the apparent singularity and
preserves gradients at \code{z = 0}. The likelihood uses a direct log-CDF
calculation with a lower-tail asymptotic expansion, avoiding cancellation and
a hard probability floor.
}
\value{
A \code{\link[torch]{torch_tensor}} of CDF values.
}
\references{
Hill, G. W. (1970).
Algorithm 395: Student's t-distribution.
\emph{Communications of the ACM}, \strong{13}, 617--619.

Brophy, A. L. (1987).
Efficient estimation of probabilities in the t distribution.
\emph{Behavior Research Methods, Instruments, & Computers},
\strong{19}, 462--466.
}
