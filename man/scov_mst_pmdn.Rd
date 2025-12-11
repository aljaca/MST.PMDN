\name{scov_mst_pmdn}
\alias{scov_mst_pmdn}
\title{Derive component covariance or scale matrices from MST-PMDN predictions}
\description{
Transforms the volume-shape-orientation decomposition returned by
\code{\link{predict_mst_pmdn}} into full covariance matrices or their Cholesky
factors for each mixture component.
}
\usage{
scov_mst_pmdn(pred, type = c("cov", "scale_chol"), as_array = FALSE)
}
\arguments{
  \item{pred}{List output from \code{\link{predict_mst_pmdn}} containing
  elements \code{L}, \code{A}, and \code{D} for the volume, diagonal shape,
  and orientation matrices.}
  \item{type}{Character; set to \code{"cov"} to return covariance matrices
  \eqn{\Sigma = L^{1/2} D \sqrt{A} (L^{1/2} D \sqrt{A})^T} or
  \code{"scale_chol"} to return their Cholesky factors.}
  \item{as_array}{Logical; if \code{TRUE}, converts the result to a base R
  array on CPU. Otherwise returns a \code{torch_tensor} on the original device.}
}
\value{
A 4D \code{torch_tensor} (or base R array if \code{as_array = TRUE}) of shape
\code{[batch_size, M, d, d]} containing either covariance matrices or their
Cholesky factors for each of the \code{M} mixture components.
}
\examples{
## Assume pred <- predict_mst_pmdn(model, newdata)
## Get covariance matrices
# cov_mats <- scov_mst_pmdn(pred, type = "cov")
## Get Cholesky factors as a base R array
# chol_arr <- scov_mst_pmdn(pred, type = "scale_chol", as_array = TRUE)
}
