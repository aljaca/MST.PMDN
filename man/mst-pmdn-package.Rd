\name{MST.PMDN-package}
\alias{MST.PMDN-package}
\docType{package}
\title{Deep Multivariate Skew t-Parsimonious Mixture Density Network}
\description{
The \pkg{MST.PMDN} package provides a distributional deep learning regression framework built on \pkg{torch} for R.
It implements a mixture of multivariate skew t distributions whose scale matrices use an
LAD eigen-decomposition (volume–shape–orientation) parameterization, and exposes high-level functions
to define, train, predict, and sample from these models.
}
\details{
Key user-facing functions include:
\itemize{
  \item \code{\link{define_mst_pmdn}}: construct a new MST-PMDN model object.
  \item \code{\link{train_mst_pmdn}}: fit an MST-PMDN model to training data.
  \item \code{\link{predict_mst_pmdn}}: generate predictive distributions or point predictions.
  \item \code{\link{loss_mst_pmdn}}: compute the negative log-likelihood loss for evaluation.
  \item \code{\link{sample_mst_pmdn}}: draw samples as \code{\link[torch]{torch_tensor}} objects.
  \item \code{\link{sample_mst_pmdn_df}}: draw samples returned as an R \code{\link{data.frame}}.
}
}
\author{
Alex J. Cannon \email{alex.cannon@canada.ca}{aut, cre}
}
\keyword{package}
