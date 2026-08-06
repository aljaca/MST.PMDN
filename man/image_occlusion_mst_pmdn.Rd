\name{image_occlusion_mst_pmdn}
\alias{image_occlusion_mst_pmdn}
\title{Spatial occlusion effects for an MST-PMDN predictive functional}
\usage{
image_occlusion_mst_pmdn(
  model,
  inputs,
  image_inputs,
  reference_images,
  functional,
  patch_size,
  stride,
  cases = NULL,
  taper = c("cosine", "none"),
  channel_groups = NULL,
  rebuild_channels = NULL,
  decompose = FALSE,
  channels = c("location", "scale", "skewness", "df"),
  num_samples = 4096L,
  latent_draws = NULL,
  seed = NULL,
  chunk_size = NULL,
  device = "cpu",
  response_names = NULL
)
}
\arguments{
  \item{model,inputs,image_inputs,reference_images,functional,cases,
  rebuild_channels,decompose,channels,num_samples,latent_draws,seed,chunk_size,
  device,response_names}{As in \code{\link{image_contrast_mst_pmdn}}.}
  \item{patch_size}{Two positive integers giving patch rows and columns. This
  is required because a scientifically appropriate spatial scale is
  application dependent.}
  \item{stride}{Two positive integers giving row and column stride. This is
  required because runtime and overlap are application dependent.}
  \item{taper}{Cosine-tapered or untapered patch replacement.}
  \item{channel_groups}{\code{NULL} to perturb all channels together, a vector
  of R 1-based channel indices, or a named list of such groups.}
}
\description{
Replaces tapered spatial patches by a reference field and reports original
minus occluded changes in a scalar predictive functional.
}
\value{
An object of class \code{mst_pmdn_image_occlusion}. It contains case-level
patch effects, population mean signed effect, mean absolute effect, positive-
effect frequency, patch geometry, spatial coverage, optional one-component
parameter-channel maps, and numerical closure residuals.
}
\details{
The callback can perturb a scientifically fundamental field, recompute linked
channels such as pressure gradients, apply original preprocessing, and return
the complete model-ready tensor. Rebuilt original and occluded states must
have matching shapes, representations, dtypes, and devices. Overlapping patch
effects are spatial sensitivity measures: parameter channels sum within each
patch, but effects generally do not sum across patches to the whole-image
contrast.
}
