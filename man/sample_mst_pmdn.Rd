\name{sample_mst_pmdn}
\alias{sample_mst_pmdn}
\title{Draw raw tensor samples from an MST-PMDN predictive mixture}

\usage{
sample_mst_pmdn(mdn_output, num_samples = 1, device = "cpu")
}

\arguments{
  \item{\code{mdn_output}}{A \code{list} as returned by \code{\link{predict_mst_pmdn}} or a forward pass of an \code{mst_pmdn_model}, containing at minimum the following named \code{\link[torch]{torch_tensor}} elements: \code{pi}, \code{mu}, \code{scale_chol}, \code{nu}, and \code{alpha}.}
  \item{\code{num_samples}}{Integer; number of random draws to generate per input case. Defaults to \code{1}.}
  \item{\code{device}}{Character; a \pkg{torch} device on which to perform sampling (e.g. \code{"cpu"} or \code{"cuda"}). Defaults to \code{"cpu"}.}
}

\description{
Performs mixture sampling from the multivariate skew t predictive distribution encapsulated in \code{mdn_output}.  
For each of the \code{num_samples} draws, a mixture component is sampled according to the mixture weights \code{pi}, and then a skew-\emph{t} variate is generated via a Gaussian plus Gamma construction.  
The output tensors are permuted so that the first dimension indexes the draw number.  
}

\value{
A \code{list} with components:
\describe{
  \item{\code{samples}}{A \code{\link[torch]{torch_tensor}} of shape \code{c(num_samples, batch_size, d)}, giving the sampled responses.}
  \item{\code{components}}{A \code{\link[torch]{torch_tensor}} of shape \code{c(num_samples, batch_size)}, containing the index (1–\code{M}) of the mixture component used for each draw.}
}
}

