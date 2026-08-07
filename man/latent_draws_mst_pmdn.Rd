\name{latent_draws_mst_pmdn}
\alias{latent_draws_mst_pmdn}
\title{Create common random numbers for MST-PMDN interpretation}
\description{
Creates a parameter-independent bank of uniforms and standard normals used to
evaluate original, perturbed, and hybrid predictive distributions with common
random numbers.
}
\usage{
latent_draws_mst_pmdn(
  num_samples = 4096L,
  output_dim,
  dtype = torch_float(),
  device = "cpu",
  seed = NULL
)
}
\arguments{
  \item{num_samples}{Positive integer number of Monte Carlo draws.}
  \item{output_dim}{Positive integer response dimension.}
  \item{dtype}{A floating-point torch dtype.}
  \item{device}{Torch device on which the bank is stored.}
  \item{seed}{Optional non-negative integer used for the torch RNG. The R RNG
  stream is not changed.}
}
\details{
The bank contains uniforms for inverse-CDF component selection, scalar normals
and multivariate normals for the skew-normal construction, and uniforms that
are transformed by \code{stats::qgamma()} for finite Student-t scaling. The
same latent sequence is shared across prediction rows. This correlates Monte
Carlo error between rows without changing any row's marginal predictive
distribution and makes case chunking reproducible. Consequently, shared
component-selection error does not average away along an ALE curve. Functional
tail-draw diagnostics describe the marginal event or quantile; mixture
diagnostics separately report expected component draw counts.
}
\value{
An object of class \code{mst_pmdn_latent_draws} containing torch tensors and
draw metadata.
}
