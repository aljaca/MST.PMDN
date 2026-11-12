\name{responsibilities_mst_pmdn}
\alias{responsibilities_mst_pmdn}
\title{Posterior Component Responsibilities for MST-PMDN}
\description{
Computes component probabilities conditional on the predictors and the
observed joint response for each prediction row.
}
\usage{
responsibilities_mst_pmdn(pred, target, log = FALSE, as_array = FALSE)
}
\arguments{
  \item{pred}{A list returned by \code{\link{predict_mst_pmdn}} or a
    compatible model forward pass. Each row of \code{pred$pi} must
    contain finite, non-negative weights with at least one positive weight.}
  \item{target}{A finite numeric matrix or real \code{\link[torch]{torch_tensor}}
    of shape \code{[batch_size, d]}. Supply the complete response on the
    fitted scale in the same row and response order as \code{pred}.
    Matrix column names are checked against the \code{response_names}
    attribute of \code{pred} when both are supplied. The target is converted
    to the device and dtype of \code{pred$mu}.}
  \item{log}{Logical. Return natural log probabilities if \code{TRUE}.}
  \item{as_array}{Logical. Return a base R numeric matrix if \code{TRUE}.}
}
\details{
For each row, component probabilities are proportional to the predicted
mixture weights times the component joint densities evaluated at
\code{target}. The calculation uses the component log densities from
\code{\link{loss_mst_pmdn}} and normalizes across components in log space.
It uses the weights in \code{pred$pi} directly. The default positive
\code{min_mix_weight} keeps fitted-model weights positive. With
\code{min_mix_weight = 0}, softmax weights can underflow to zero.
Hand-built predictions can also specify zeros. A component with zero
weight has probability zero and log probability \code{-Inf}. The training
loss applies a separate floor of \eqn{10^{-12}} to its weights.

The component densities inherit the likelihood's numerical guards and
log-CDF approximations. In particular, squared Mahalanobis distances
above \eqn{10^6} are capped. This can change responsibilities relative
to the uncapped densities. The finite-t density also caps the ratio of
the squared distance to the degrees of freedom at \eqn{10^7}, which can
bind for fixed degrees of freedom below 0.1. The finite-t skew multiplier
is capped at \eqn{10^6}, which can bind for extremely small fixed degrees
of freedom. For the underlying densities, a common invertible affine
transformation of the response and all component distributions leaves
the responsibilities unchanged because the common Jacobian cancels.

Tensor output preserves the caller's autograd context. To match
\code{predict_mst_pmdn} while tracking gradients to the model parameters,
call \code{model$eval()} and pass the result of a gradient-enabled model
forward pass as \code{pred}. For differentiable entropy, use log
responsibilities and replace \code{-Inf} with zero in the log factor before
multiplying by the probabilities. Inactive components contribute zero.
}
\value{
A \code{\link[torch]{torch_tensor}} of shape \code{[batch_size, M]} on the
prediction device, or a base R matrix of the same shape when
\code{as_array = TRUE}. Column \eqn{k} corresponds to fitted component
\eqn{k}. Probability rows sum to one up to floating-point precision.
}
\seealso{
\code{\link{predict_mst_pmdn}}, \code{\link{loss_mst_pmdn}}
}
