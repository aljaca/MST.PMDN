\name{scov_mst_pmdn}
\alias{scov_mst_pmdn}
\title{Derive component scale or covariance matrices from MST-PMDN predictions}
\description{
Returns full scale matrices, their Cholesky factors, or actual component
covariance matrices from \code{\link{predict_mst_pmdn}} output.
}
\usage{
scov_mst_pmdn(pred, type = c("scale", "scale_chol", "cov"), as_array = FALSE)
}
\arguments{
  \item{pred}{List output from \code{\link{predict_mst_pmdn}} containing
  \code{scale_chol}, \code{nu}, and \code{alpha}.}
  \item{type}{Character; \code{"scale"} returns the skew-t scale matrix,
  \code{"scale_chol"} returns its Cholesky factor, and \code{"cov"} returns
  the component covariance after accounting for degrees of freedom and
  skewness. Component covariance is undefined for finite \eqn{\nu \le 2};
  \code{nu = Inf} uses the exact skew-normal limit.}
  \item{as_array}{Logical; if \code{TRUE}, converts the result to a base R
  array on CPU. Otherwise returns a \code{torch_tensor} on the original device.}
}
\value{
A 4D \code{torch_tensor} (or base R array if \code{as_array = TRUE}) of shape
\code{[batch_size, M, d, d]} containing scale matrices, Cholesky factors, or
component covariance matrices. Entries for \code{type = "cov"} are \code{NaN}
where \eqn{\nu \le 2}.
}
\details{
Let \eqn{C} denote \code{scale_chol},
\eqn{\delta = \alpha / \sqrt{1 + \alpha^T\alpha}}, and
\deqn{b_\nu = \sqrt{\nu/\pi}\,\Gamma((\nu-1)/2)/\Gamma(\nu/2).}
For \eqn{\nu>2}, the component covariance is
\deqn{C\left[\frac{\nu}{\nu-2}I-b_\nu^2\delta\delta^T\right]C^T.}
For \eqn{\nu=\infty}, this becomes the exact skew-normal covariance
\deqn{C\left[I-\frac{2}{\pi}\delta\delta^T\right]C^T.}
The default \code{type = "scale"} retains the numerical default returned by
versions prior to 0.2.0, where it was incorrectly labelled covariance.
}
