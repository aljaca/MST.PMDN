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
- Added regression tests and an R CMD check workflow.
