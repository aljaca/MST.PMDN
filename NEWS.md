# MST.PMDN 0.2.0

- Corrected one-based mixture-component sampling and the multivariate
  skew-normal construction used by both sampling helpers.
- Corrected the empirical marginal CDF estimator.
- Distinguished skew-t scale matrices from component covariance matrices and
  documented `mu` as a location parameter.
- Preserved the intended neutral initialization of learned `nu` and `alpha`
  heads.
- Made checkpoints fully resumable by restoring optimizer, split, history,
  counters, and available R/torch RNG state; latest and best states now use
  separate files.
- Validation now evaluates every case and reports observation-weighted losses.
- Replaced the Student t CDF approximation with the more accurate
  Hill transformation and a stable direct log-CDF, preserving gradients at
  zero and removing the lower-tail probability floor from the likelihood.
- Made the degrees-of-freedom `"N"` constraint an exact Gaussian/skew-normal
  limit and allowed `Inf` in `fixed_nu` for exact normal components within
  mixed fixed/learned models.
- Made floating-point tensors created in the model forward pass inherit the
  input dtype, including mixed fixed/learned degrees-of-freedom models.
- Reduced the default learned degrees-of-freedom range from `c(3, 500)` to
  `c(3, 50)` and stabilized the multivariate t normalizing constant against
  float32 gamma-function cancellation.
- Removed the unused schema version from pre-release checkpoint payloads.
- Added regression tests and an R CMD check workflow.
