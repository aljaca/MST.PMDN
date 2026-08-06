# MST.PMDN 0.3.0

## Distribution-functional interpretation

- Added validated scalar functional specifications and evaluators for exact
  means, variances, standard deviations, covariances, and correlations, plus
  Monte Carlo quantiles, marginal and joint exceedances, tail spread, and tail
  asymmetry. Exact mixture covariance includes between-component variation and
  undefined low-degrees-of-freedom moments are explicit.
- Added parameter-independent latent banks with common random numbers for
  component selection, the skew-normal construction, and Gamma-quantile
  Student-t scaling. Fixed banks preserve dtype and make chunked and unchunked
  functional evaluation identical.
- Tail summaries now report expected tail-draw counts and warn when Monte Carlo
  resolution is inadequate. Counts use untransformed probabilities, and ALE
  diagnostics summarize the states actually evaluated rather than a nominal
  probability.
- Exact-Gaussian degrees-of-freedom diagnostics retain `Inf` without an
  undefined-moment warning.

## Covariate and image effects

- Added one-dimensional accumulated local effects, centred ICE curves, local
  finite-difference slopes, and Plate-style baseline-contrast/slope data.
  Case-matched image rows remain fixed during tabular perturbations.
- Added whole-image functional contrasts and tapered spatial occlusion maps,
  with signed, absolute, and sign-consistency population summaries.
- Added a `rebuild_channels` callback contract for perturbing fundamental
  physical fields and rebuilding deterministically linked image channels before
  model evaluation. Reference images are aligned to the source representation,
  dtype, and device, and rebuilt states are checked for mutual compatibility.

## Distributional attribution

- Added exact Shapley decomposition of one-component functional contrasts among
  complete location, scale, skewness, and degrees-of-freedom blocks. Inactive
  blocks disappear and every result reports its sum-to-total residual.
- Added mixture-safe exceedance accounting through component weight,
  within-component event probability, weighted contribution, tail share, and
  contribution rank. Full parameter-channel attribution remains disabled for
  mixtures.
- Added base-R S3 plotting methods, manual pages, wave-surge and synthetic
  workflow examples, and CPU/float32/float64/conditional-CUDA regression tests.

# MST.PMDN 0.2.1

## Bug fixes

- Sampler latent-normal and Gamma tail-scale draws now follow the model dtype.
  Float64 models previously drew float32 latents and relied on implicit
  promotion. Gamma draws are now made in one vectorized R call.
- `predict_mst_pmdn()` now coerces non-tensor tabular and image inputs to the
  model dtype instead of always using float32. Supplied tensors retain their
  dtype, and a mismatch with the model now errors at the call boundary rather
  than inside the network, with an explicit conversion remedy. Parameterless
  modules retain the previous float32 coercion for non-tensor inputs and
  preserve the dtype of supplied tensors.

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
