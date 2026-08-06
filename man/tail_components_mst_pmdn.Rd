\name{tail_components_mst_pmdn}
\alias{tail_components_mst_pmdn}
\title{Attribute an MST-PMDN mixture exceedance probability to components}
\description{
Separates component prevalence, within-component event propensity, and weighted
contribution to a complete mixture exceedance probability.
}
\usage{
tail_components_mst_pmdn(
  pred,
  response,
  threshold,
  direction = "upper",
  num_samples = 4096L,
  latent_draws = NULL,
  seed = NULL,
  chunk_size = NULL,
  device = "cpu",
  response_names = NULL,
  min_tail_draws = 20L
)
}
\arguments{
  \item{pred}{MST-PMDN prediction list with one or more components.}
  \item{response}{Response name or R 1-based index.}
  \item{threshold}{Finite event threshold.}
  \item{direction}{\code{"upper"} or \code{"lower"}.}
  \item{num_samples}{Monte Carlo draw count.}
  \item{latent_draws}{Optional common latent bank.}
  \item{seed}{Optional latent-bank seed.}
  \item{chunk_size}{Optional case chunk size.}
  \item{device}{Torch evaluation device.}
  \item{response_names}{Optional prediction response names.}
  \item{min_tail_draws}{Warning threshold for expected mixture-tail draws.}
}
\details{
For component \eqn{g}, the output reports \eqn{p_g=P_g(Y>u)}, weighted
contribution \eqn{r_g=\pi_g p_g}, and tail share
\eqn{\rho_g=r_g/\sum_h r_h}. Contribution ranks and the unordered set of
contributions are invariant to component-label permutations. Quantiles have no
corresponding additive component decomposition.
}
\value{
An \code{mst_pmdn_tail_components} object with one data-frame row per
prediction row and component, plus settings, diagnostics, and the latent bank.
}
