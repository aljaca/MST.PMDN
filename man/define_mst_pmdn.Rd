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
  fusion_module   = NULL,
  fixed_nu        = NULL,
  range_nu        = c(3, 50),
  max_alpha       = 2.5,
  min_vol_shape   = 1e-2,
  min_mix_weight  = 1e-4,
  jitter          = 1e-6
)
}
\arguments{
  \item{input_dim}{Integer. Number of features in the raw tabular input (used if \code{tabular_module} is \code{NULL}).}
  \item{output_dim}{Integer. Dimension \code{d} of the multivariate response.}
  \item{hidden_dim}{Integer vector. Layer widths in the default dense MLP. When \code{fusion_module} is supplied, these values create no layers; the final value is used only if the fusion output dimension cannot be inferred.}
  \item{n_mixtures}{Integer. Number of skew t mixture components \code{M}.}
  \item{constraint}{Character. Code for volume–shape–orientation, degrees of freedom \code{nu}, and skew constraints (e.g., “\code{VVVFN}”). An \code{"N"} degrees-of-freedom constraint selects the exact Gaussian/skew-normal limit.}
  \item{constant_attr}{Character. Flags for parameters held constant (e.g., “\code{m}” for skew-t locations, “\code{s}” for skew, etc.).}
  \item{activation}{Function or list of functions. \pkg{torch} activation applied after each non-final layer of the default dense MLP (default \code{nn_relu}). Ignored when \code{fusion_module} is supplied.}
  \item{drop_hidden}{Numeric between 0 and 1. Dropout probability after each non-final layer of the default dense MLP. When \code{fusion_module} is supplied, dropout is applied once to the fusion output.}
  \item{image_module}{A \pkg{torch} \code{nn_module} for image feature extraction, or \code{NULL} to omit.}
  \item{tabular_module}{A \pkg{torch} \code{nn_module} for tabular feature extraction, or \code{NULL} to use raw inputs.}
  \item{fusion_module}{Optional \pkg{torch} \code{nn_module} that fuses tabular and image features. When supplied, it bypasses the default dense MLP; after optional \code{drop_hidden} dropout, its output is passed directly to the parameter-prediction heads. The output width is inferred from \code{output_dim}, \code{out_features}, \code{out_channels}, or \code{out_dim}; the final \code{hidden_dim} value is used as a fallback.}
  \item{fixed_nu}{Numeric vector of length \code{M}, or \code{NULL}. Under an \code{"F"} degrees-of-freedom constraint, each entry may be a positive finite value, \code{Inf} for an exact Gaussian/skew-normal component, or \code{NA} for a learned finite-\code{nu} component.}
  \item{range_nu}{Two increasing positive finite values, \code{c(min_nu, max_nu)}, bounding learned \code{nu}. Defaults to \code{c(3, 50)} so initialization and most of the sigmoid range remain in the statistically informative finite-tail regime.}
  \item{max_alpha}{Numeric. Absolute bound for skewness parameter.}
  \item{min_vol_shape}{Numeric. Minimum allowed value for volume (L) and shape (A) diagonal parameters.}
  \item{min_mix_weight}{Numeric. Floor for each mixture weight to avoid zero probabilities.}
  \item{jitter}{Numeric. Small ridge added to covariance diagonals for numerical stability.}
}
\value{
An untrained \code{mst_pmdn_model}, which is a \pkg{torch} \code{nn_module} encapsulating the MST‐PMDN head.  Use \code{\link{train_mst_pmdn}} to fit it to data.
}
