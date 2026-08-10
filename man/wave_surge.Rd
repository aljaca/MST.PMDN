\name{wave_surge}
\alias{wave_surge}
\title{Daily Maximum Wave and Surge Data}
\docType{data}
\usage{data(wave_surge)}
\description{
A model-ready set of 896 daily maximum significant-wave-height and
storm-surge observations from CCCRIS node 181947 at Roberts Bank, with
seasonal and lagged-response predictors and collocated ERA5 atmospheric
fields.
}
\format{
A named \code{list} with 11 components:
\describe{
  \item{date}{Character vector of length 896 containing ISO-formatted dates.
  Its entries align with the first dimension or rows of \code{x},
  \code{x_image}, and \code{y}.}
  \item{y}{Numeric matrix with 896 rows and two columns,
  \code{softplus_Wave.m} and \code{Surge.m}. Significant wave height is
  mapped to a real-valued latent scale with \code{inv_softplus()} before
  standardization; surge is standardized directly.}
  \item{x}{Numeric matrix with 896 rows and 10 columns containing the first
  two sine/cosine annual harmonics and lagged wave and surge predictors.
  Predictor definitions and ordering are retained in \code{colnames(x)}.}
  \item{x_image}{Numeric array with dimensions 896 by 3 by 32 by 32. The
  dimensions index case, channel, and the two spatial coordinates. The
  standardized channels, in order, are mean sea-level pressure
  (\code{psl}), eastward near-surface wind (\code{uas}), and northward
  near-surface wind (\code{vas}).}
  \item{y_mean}{Named numeric vector of length two containing the response
  means used before standardization, on the inverse-softplus wave scale and
  the original surge scale.}
  \item{y_sd}{Named numeric vector of length two containing the response
  standard deviations used for standardization.}
  \item{x_image_mean}{Numeric array with dimensions 3 by 32 by 32 containing
  the grid-cell-specific, original-unit means used to standardize
  \code{x_image}. Its channel order matches \code{x_image}.}
  \item{x_image_sd}{Numeric array with dimensions 3 by 32 by 32 containing
  the grid-cell-specific, original-unit standard deviations used to
  standardize \code{x_image}. Its channel order matches \code{x_image}.}
  \item{wave_scale}{Numeric scalar equal to 0.1: the scale used by the
  significant-wave-height softplus transformation.}
  \item{inv_softplus}{Function with arguments \code{x} and
  \code{x_scale = 0.1} that maps positive significant wave height to the
  real-valued latent wave scale.}
  \item{softplus}{Function with arguments \code{x} and
  \code{x_scale = 0.1} that maps the latent wave scale back to nonnegative
  significant wave height.}
}
}
\details{
The image predictors were standardized separately at every channel and grid
cell, where \eqn{\mu} and \eqn{\sigma} are stored in
\code{x_image_mean} and \code{x_image_sd}:
\deqn{
x^{\mathrm{std}}_{t,c,i,j} =
\frac{x^{\mathrm{raw}}_{t,c,i,j} - \mu_{c,i,j}}
{\sigma_{c,i,j}}.
}

For a standardized first-response value \code{z}, significant wave height is
recovered with
\code{softplus(z * y_sd[1] + y_mean[1], wave_scale)}. For a standardized
second-response value \code{z}, surge is recovered with
\code{z * y_sd[2] + y_mean[2]}. The inverse response transformation uses
\code{inv_softplus()} before centering and scaling wave height.
}
\source{
CCCRIS node 181947 (\url{https://cccris.ca}) and ERA5 reanalysis.
}
\keyword{datasets}
