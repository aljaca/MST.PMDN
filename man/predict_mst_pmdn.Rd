\name{predict_mst_pmdn}
\alias{predict_mst_pmdn}
\title{Generate predictions from a trained MST-PMDN model}
\description{
Wraps a trained \code{mst_pmdn_model} in evaluation mode, converts inputs to
\code{\link[torch]{torch_tensor}} if necessary using the model's floating-point
dtype, and runs a forward pass without tracking gradients to produce
mixture-distribution parameters.
}
\usage{
predict_mst_pmdn(model, new_inputs, image_inputs = NULL, device = "cpu")
}
\arguments{
  \item{model}{An \code{mst_pmdn_model} object (a \pkg{torch} module), typically returned by \code{\link{train_mst_pmdn}}.}
  \item{new_inputs}{A numeric matrix or \code{\link[torch]{torch_tensor}} of tabular predictors. If not already a tensor, it is converted on the specified \code{device} using the dtype of the model's first parameter, or float32 when the model has no parameters. A supplied tensor retains its dtype and must match the model dtype when one can be inferred.}
  \item{image_inputs}{Optional 4D array or \code{torch_tensor} of image data for models with an \code{image_module}. Non-tensor input is converted to the model dtype, or float32 when the model has no parameters. A supplied tensor retains its dtype and must match the model dtype when one can be inferred. If provided, the input must match the module's expected shape; otherwise set to \code{NULL}.}
  \item{device}{Character; the torch device to use (e.g. \code{"cpu"} or \code{"cuda"}).  Defaults to \code{"cpu"}.}
}
\value{
A named \code{list} containing the output of the model’s forward pass:
\describe{
  \item{\code{pi}}{Mixture weights tensor of shape \code{[batch_size, M]}.}
  \item{\code{mu}}{Component skew-t location tensor of shape \code{[batch_size, M, d]}. This is not the component mean when skewness is nonzero.}
  \item{\code{scale_chol}}{Cholesky scale factors tensor of shape \code{[batch_size, M, d, d]}.}
  \item{\code{nu}}{Degrees-of-freedom tensor of shape \code{[batch_size, M]}; \code{Inf} marks an exact Gaussian/skew-normal component.}
  \item{\code{alpha}}{Skewness parameters tensor of shape \code{[batch_size, M, d]}.}
  \item{\code{skew_none}}{A single logical value indicating whether the model
  has the \code{"N"} skewness constraint. Older or hand-built output lists may
  omit this field; consumers then evaluate the general skew path.}
  \item{\code{L}, \code{A}, \code{D}}{Volume, shape, and orientation decomposition tensors for each component.}
}
}
\details{
The model dtype is inferred from its first parameter. For a model with no
parameters, non-tensor inputs retain the legacy float32 conversion and supplied
tensors retain their dtype; no model-dtype validation is possible. For a
parameterized model, tensor inputs with a different dtype produce an error
before the model's first matrix multiplication. The diagnostic names the
parameter used for inference (or reports \code{<unnamed>}) and shows how to
convert the input explicitly with \code{$to(dtype = ...)}.
}
