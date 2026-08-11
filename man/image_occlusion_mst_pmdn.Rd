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
  response_names = NULL,
  min_tail_draws = 20L
)
}
\arguments{
  \item{model,inputs,image_inputs,reference_images,functional,cases,
  rebuild_channels,decompose,channels,num_samples,seed,chunk_size,
  device,response_names,min_tail_draws}{As in
  \code{\link{image_contrast_mst_pmdn}}.}
  \item{latent_draws}{One common latent bank, or a non-empty list of
  independent banks with equal draw counts. Patch predictions are computed
  once and reused across banks.}
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
patch effects and bank-replicate effects, population summaries, between-bank
spread and sign-stability diagnostics, patch geometry and taper weights,
binary and weighted spatial coverage, optional one-component parameter-channel
maps, and numerical closure residuals.
}
\details{
The callback can perturb a scientifically fundamental field, recompute linked
channels such as pressure gradients, apply original preprocessing, and return
the complete model-ready tensor. With \code{channel_groups = NULL}, callback
masks have shape \code{[case, 1, row, column]}; grouped masks have the full
\code{[case, channel, row, column]} shape. Rebuilt original and occluded states
must have matching shapes, representations, dtypes, and devices.

Cosine tapering changes the input perturbation and is not interpreted as a
fixed fraction of full occlusion. Raw effects are retained; \code{mask_sum},
\code{mask_mean}, and \code{mask_max} describe patch strength. Binary
\code{coverage} counts overlapping patch rectangles, while
\code{weighted_coverage} sums taper weights.

With multiple latent banks, \code{effect} is their mean and replicate values
are retained separately. A sign is marked stable only when every bank-specific
effect lies strictly on the same side of zero. This is a Monte Carlo
sign-stability heuristic, not a confidence interval. Overlapping patch effects
are spatial sensitivity measures: parameter channels sum within each patch,
but effects generally do not sum across patches to the whole-image contrast.
}
