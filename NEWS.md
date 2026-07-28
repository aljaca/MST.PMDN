# MST.PMDN 0.2.1

## Performance

- Models with the `"N"` skewness constraint now bypass the skew-factor block
  in `loss_mst_pmdn()` and the skew-normal construction in
  `sample_mst_pmdn()`. Loss values and trainable-parameter gradients are
  unchanged. Inactive skewness and degrees-of-freedom penalties no longer
  construct reduction graphs.
- Symmetric sampling consumes a different RNG sequence because the redundant
  scalar-normal draw is no longer made. Fixed seeds remain reproducible within
  this version but do not reproduce sample streams from earlier versions.

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
- Made floating-point tensors created in the model forward pass and loss
  calculations inherit the model dtype, including mixed fixed/learned
  degrees-of-freedom models.
- Reduced the default learned degrees-of-freedom range from `c(3, 500)` to
  `c(3, 50)` and stabilized the multivariate t normalizing constant against
  float32 gamma-function cancellation.
- Removed the unused schema version from pre-release checkpoint payloads.
- Added regression tests and an R CMD check workflow.
