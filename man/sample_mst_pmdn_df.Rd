\name{sample_mst_pmdn_df}
\alias{sample_mst_pmdn_df}
\title{Draw MST-PMDN predictive samples as a data frame}
\docType{function}

\usage{
sample_mst_pmdn_df(mdn_output, num_samples = 1, device = "cpu")
}

\arguments{
  \item{\code{mdn_output}}{A \code{list} as returned by \code{\link{predict_mst_pmdn}} or a forward pass of an \code{mst_pmdn_model}, containing at minimum the following named \code{\link[torch]{torch_tensor}} elements: \code{pi}, \code{mu}, \code{scale_chol}, \code{nu}, and \code{alpha}.}
  \item{\code{num_samples}}{Integer; number of random draws to generate per input case. Defaults to \code{1}.}
  \item{\code{device}}{Character; the torch device on which sampling is performed (e.g.\ \code{"cpu"} or \code{"cuda"}). Defaults to \code{"cpu"}.}
}

\description{
For each of \code{num_samples} draws per case, a mixture component is sampled according to the weights \code{pi}, and then a multivariate skew-\emph{t} variate is generated via the usual Gaussian + Gamma construction.  
The results are returned in long format as an R \code{\link{data.frame}} with one row per draw, including the sampled values for each response dimension, the input case index, the draw index, and the component chosen.
}

\value{
A \code{\link{data.frame}} with \code{num_samples * B} rows (where \code{B =} number of input cases) and \code{d + 3} columns:
\describe{
  \item{\code{V1,\dots,Vd}}{Numeric; sampled values for each of the \emph{d} response dimensions.}
  \item{\code{row}}{Integer; index of the input case (1 through \code{B}).}
  \item{\code{draw}}{Integer; draw number (1 through \code{num_samples}).}
  \item{\code{comp}}{Factor; mixture component index (1 through \code{M}) used for that draw.}
}
}
