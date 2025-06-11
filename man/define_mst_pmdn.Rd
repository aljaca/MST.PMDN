\name{define_mst_pmdn}
\alias{define_mst_pmdn}
\title{Construct a Deep Multivariate Skew t-Parsimonious Mixture Density Network}
\description{
Initializes an untrained deep MST-PMDN model as a \pkg{torch} module, ready for training with \code{\link{train_mst_pmdn}}.  
The model predicts parameters of a mixture of multivariate skew t distributions using a fusion of optional tabular and image feature extractors.  
}
\usage{
define_mst_pmdn(input_dim,
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
                jitter          = 1e-6)
\arguments{
  \item{\code{input_dim}}{Integer scalar. Number of features in the tabular input (or fallback dimension if \code{tabular_module} is \code{NULL}).}
  \item{\code{output_dim}}{Integer scalar. Dimension \code{d} of the multivariate response.}
  \item{\code{hidden_dim}}{Integer vector. Sizes of each hidden layer in the dense fusion network.}
  \item{\code{n_mixtures}}{Integer scalar. Number of mixture components \code{M}.}
  \item{\code{constraint}}{Character string. MST-PMDN constraint code (volume–shape–orientation, skew, and nu parameters), e.g.\ \code{"VVVNN"}.}
  \item{\code{constant_attr}}{Character string. Flags for which parameters are held constant (see package vignette for details), e.g.\ \code{""}, \code{"m"} (constant means), \code{"s"} (constant skew), or combinations.}
  \item{\code{activation}}{Activation function for hidden layers (a \pkg{torch} function), default \code{nn_relu}.}
  \item{\code{drop_hidden}}{Numeric in [0,1]. Dropout probability after each hidden layer, default 0 (no dropout).}
  \item{\code{image_module}}{A \pkg{torch} module (class \code{nn_module}) to extract image features, or \code{NULL} to omit image inputs.}
  \item{\code{tabular_module}}{A \pkg{torch} module to extract tabular features, or \code{NULL} to use raw inputs.}
  \item{\code{fixed_nu}}{Numeric vector of length \code{M}, or \code{NULL}. If non-\code{NULL}, fixes degrees of freedom for each mixture component.}
  \item{\code{range_nu}}{Numeric vector of length 2: \code{c(min_nu, max_nu)}. Clamps any learned \code{nu} parameter into this range.}
  \item{\code{max_alpha}}{Numeric scalar. Absolute bound for the skewness parameter \(\alpha\), so that \(\alpha\in[-\max\alpha,\max\alpha]\).}
  \item{\code{min_vol_shape}}{Numeric scalar. Minimum allowed value for the volume (L) and shape (A) diagonal parameters, ensuring well-conditioned covariance matrices.}
  \item{\code{min_mix_weight}}{Numeric scalar. Minimum mixture weight (floor) for each component to avoid zero-weight issues.}
  \item{\code{jitter}}{Numeric scalar. Small ridge added to diagonal of covariance before Cholesky, for numerical stability.}
}
\value{
An object of class \code{nn_module} implementing the specified MST-PMDN head.  
The returned module has methods for forward inference and parameter access but must be trained via \code{\link{train_mst_pmdn}}.}
\examples{
library(torch)

## Define a 2-component model for 3-dimensional outputs with a 2-layer fusion net
model <- define_mst_pmdn(
  input_dim      = 8,
  output_dim     = 3,
  hidden_dim     = c(64, 32),
  n_mixtures     = 2,
  constraint     = "VVVNN",
  activation     = nn_relu,
  drop_hidden    = 0.1,
  range_nu       = c(3, 40),
  max_alpha      = 3,
  min_vol_shape  = 1e-2,
  min_mix_weight = 1e-3,
  jitter         = 1e-6
)
}
