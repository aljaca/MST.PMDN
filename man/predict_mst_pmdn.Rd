\name{predict_mst_pmdn}
\alias{predict_mst_pmdn}
\title{Generate predictions from a trained MST-PMDN model}
\description{
Wraps a trained \code{mst_pmdn_model} in evaluation mode, converts inputs to 
\code{\link[torch]{torch_tensor}} if necessary, and runs a forward pass without 
tracking gradients to produce mixture-distribution parameters.
}
\usage{
predict_mst_pmdn(model, new_inputs, image_inputs = NULL, device = "cpu")
}
\arguments{
  \item{model}{An \code{mst_pmdn_model} object (a \pkg{torch} module), typically returned by \code{\link{train_mst_pmdn}}.}
  \item{new_inputs}{A numeric matrix or \code{\link[torch]{torch_tensor}} of tabular predictors.  If not already a tensor, it is converted via \code{\link[torch]{torch_tensor}} on the specified \code{device}.}
  \item{image_inputs}{Optional 4D array or \code{torch_tensor} of image data for models with an \code{image_module}.  If provided, must match the module’s expected shape; otherwise set to \code{NULL}.}
  \item{device}{Character; the torch device to use (e.g. \code{"cpu"} or \code{"cuda"}).  Defaults to \code{"cpu"}.}
}
\value{
A named \code{list} containing the output of the model’s forward pass:
\describe{
  \item{\code{pi}}{Mixture weights tensor of shape \code{[batch_size, M]}.}
  \item{\code{mu}}{Component means tensor of shape \code{[batch_size, M, d]}.}
  \item{\code{scale_chol}}{Cholesky scale factors tensor of shape \code{[batch_size, M, d, d]}.}
  \item{\code{nu}}{Degrees-of-freedom tensor of shape \code{[batch_size, M]}.}
  \item{\code{alpha}}{Skewness parameters tensor of shape \code{[batch_size, M, d]}.}
  \item{\code{L}, \code{A}, \code{D}}{Volume, shape, and orientation decomposition tensors for each component.}
}
}
