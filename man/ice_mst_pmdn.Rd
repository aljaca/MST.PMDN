\name{ice_mst_pmdn}
\alias{ice_mst_pmdn}
\title{Centred individual conditional expectation for MST-PMDN functionals}
\usage{
ice_mst_pmdn(
  model,
  inputs,
  feature,
  functional,
  image_inputs = NULL,
  grid = NULL,
  reference = NULL,
  n_curves = 100L,
  derivative = FALSE,
  ale = TRUE,
  n_bins = 20L,
  num_samples = 4096L,
  latent_draws = NULL,
  seed = NULL,
  chunk_size = NULL,
  device = "cpu",
  response_names = NULL
)
}
\arguments{
  \item{model,inputs,feature,functional,image_inputs,num_samples,latent_draws,
  seed,chunk_size,device,response_names}{As in
  \code{\link{ale_mst_pmdn}}.}
  \item{grid}{Optional increasing feature grid. The default is 25 empirical
  quantiles from 0.05 through 0.95.}
  \item{reference}{Feature value at which every curve is centred. The default
  is the empirical median.}
  \item{n_curves}{Maximum number of evenly spaced input cases.}
  \item{derivative}{If \code{TRUE}, calculate local finite-difference slopes.}
  \item{ale}{\code{TRUE} to compute an ALE overlay using all input rows,
  \code{FALSE} or \code{NULL} to omit it, or a matching precomputed
  \code{mst_pmdn_ale} object.}
  \item{n_bins}{Number of bins for the accompanying ALE overlay.}
}
\description{
Calculates complete centred ICE curves, an optional ALE overlay, local slopes,
and Plate-style baseline-contrast and local-slope data.
}
\value{
An object of class \code{mst_pmdn_ice} with plain \code{curves} and
\code{plate} data frames plus an optional \code{mst_pmdn_ale} object.
}
\details{
The same latent bank is used at every grid point. Fixed-reference ICE is
intended to show effect heterogeneity and interactions; unlike ALE, it can
construct predictor combinations with weak empirical support.
The default overlay requires a separate full-data ALE calculation; it can be
omitted or supplied precomputed when only selected ICE cases are needed.
}
