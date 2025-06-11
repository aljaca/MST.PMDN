\name{define_mst_pmdn}
\alias{define_mst_pmdn}
\title{Construct an Untrained MST‐PMDN Model}
\description{
Initializes a deep Multivariate Skew t-Parsimonious Mixture Density Network (MST‐PMDN) as a \pkg{torch} module, ready for fitting via \code{\link{train_mst_pmdn}}.  
The model predicts parameters of a mixture of multivariate skew t distributions using optional tabular and image feature extractors.
}
\usage{
define_mst_pmdn(
  input_dim,
  output_dim,
  hidden_dim,
  n_mixtures,
  constraint      = "VVVNN",
  constant_attr   = "",
  activation      = nn_relu,
  drop_hidden     = 0,
  image_module    = NULL,
  tabular_module  = NULL,
  fixed_nu        = NULL,
  range_nu        = c(3, 50),
  max_alpha       = 5,
  min_vol_shape   = 1e-2,
  min_mix_weight  = 1e-4,
  jitter          = 1e-6
)
}
\arguments{
  \item{input_dim}{Integer. Number of features in the raw tabular input (used if \code{tabular_module} is \code{NULL}).}
  \item{output_dim}{Integer. Dimension \code{d} of the multivariate response.}
  \item{hidden_dim}{Integer vector. Sizes of hidden layers in the dense fusion network.}
  \item{n_mixtures}{Integer. Number of skew t mixture components \code{M}.}
  \item{constraint}{Character. Code for volume–shape–orientation, degrees of freedom \code{nu}, and skew constraints (e.g.\ “\code{VVVFN}”).}
  \item{constant_attr}{Character. Flags for parameters held constant (e.g.\ “\code{m}” for means, “\code{s}” for skew, etc.).}
  \item{activation}{Function. \pkg{torch} activation for hidden layers (default \code{nn_relu}).}
  \item{drop_hidden}{Numeric between 0 and 1. Dropout probability after each hidden layer.}
  \item{image_module}{A \pkg{torch} \code{nn_module} for image feature extraction, or \code{NULL} to omit.}
  \item{tabular_module}{A \pkg{torch} \code{nn_module} for tabular feature extraction, or \code{NULL} to use raw inputs.}
  \item{fixed_nu}{Numeric vector of length \code{M}, or \code{NULL}. If non-\code{NULL}, fixes degrees of freedom for each component.}
  \item{range_nu}{Numeric vector of length 2: \code{c(min_nu, max_nu)} to clamp learned \code{nu}.}
  \item{max_alpha}{Numeric. Absolute bound for skewness parameter.}
  \item{min_vol_shape}{Numeric. Minimum allowed value for volume (L) and shape (A) diagonal parameters.}
  \item{min_mix_weight}{Numeric. Floor for each mixture weight to avoid zero probabilities.}
  \item{jitter}{Numeric. Small ridge added to covariance diagonals for numerical stability.}
}
\value{
An untrained \code{mst_pmdn_model}, which is a \pkg{torch} \code{nn_module} encapsulating the MST‐PMDN head.  Use \code{\link{train_mst_pmdn}} to fit it to data.
}
