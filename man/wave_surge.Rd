\name{wave_surge}
\alias{wave_surge}
\title{Wave–Surge Daily‐Maximum Subset}
\docType{data}
\usage{data(wave_surge)}
\description{
A subset of the CCCRIS node 181947 daily maximum wave and surge data and covariates.
}
\format{
A named \code{list} with three components:
\describe{
  \item{x}{Numeric array of tabular predictors (harmonics of the seasonal cyle and lag-1 wave/surge data).}
  \item{x_image}{Numeric array of image predictors (sea level pressure and sea level pressure gradient).}
  \item{y}{Numeric matrix of scaled responses (daily maximum significant wave height and surge).}
}
}
\source{
CCCRIS node 181947 (cccris.ca) and ERA5 reanalysis
}
\keyword{datasets}
