\name{image_contrast_mst_pmdn}
\alias{image_contrast_mst_pmdn}
\title{Whole-image contrasts for MST-PMDN predictive functionals}
\description{
Computes whole-image functional contrasts against required case-matched
reference fields.
}
\usage{
image_contrast_mst_pmdn(
  model,
  inputs,
  image_inputs,
  reference_images,
  functional,
  cases = NULL,
  rebuild_channels = NULL,
  decompose = FALSE,
  channels = c("location", "scale", "skewness", "df"),
  num_samples = 4096L,
  latent_draws = NULL,
  seed = NULL,
  chunk_size = NULL,
  device = "cpu",
  response_names = NULL,
  min_tail_draws = 20L
)
}
\arguments{
  \item{model}{A fitted MST-PMDN torch module.}
  \item{inputs}{Case-matched numeric tabular inputs.}
  \item{image_inputs}{Four-dimensional array or tensor ordered case, channel,
  row, column. With a callback these may be physical source fields.}
  \item{reference_images}{One reference image or one per case, matching source
  channels and spatial dimensions.}
  \item{functional}{Object returned by \code{\link{mst_functional}}.}
  \item{cases}{Optional R 1-based subset of cases.}
  \item{rebuild_channels}{Optional function with arguments
  \code{base_images}, \code{reference_images}, \code{masks}, and
  \code{case_index}. It must return complete model-ready images.}
  \item{decompose}{Whether to return one-component parameter-channel effects.}
  \item{channels}{Requested parameter channels.}
  \item{num_samples}{Monte Carlo draw count.}
  \item{latent_draws}{Optional common latent bank.}
  \item{seed}{Optional latent-bank seed.}
  \item{chunk_size}{Optional prediction and functional case chunk size.}
  \item{device}{Torch evaluation device.}
  \item{response_names}{Optional prediction response names.}
  \item{min_tail_draws}{Positive warning threshold for expected tail draws.}
}
\details{
The contrast is the functional under the observed image minus the functional
under the reference image, with tabular inputs held fixed. A callback can
rebuild deterministic channels such as pressure gradients and apply the
original preprocessing before either image is evaluated. Reference fields are
coerced to the input image representation, dtype, and device before masking.
All callback evaluations must return mutually compatible model-ready images
with the same shape, representation, dtype, and device. Monte Carlo
diagnostics from both endpoints, or all Shapley hybrids, are summarized once
per public call.
}
\value{
An \code{mst_pmdn_image_contrast} object with case contrasts, settings, and
diagnostics.
}
