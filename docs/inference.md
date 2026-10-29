# Inference and applicability

`TurboPDZ.load` accepts a combined directory with `zml/` and `sigmaml/`, or a zML
stage alone. Loading verifies file hashes, native grids, and the sigmaML parent
model identity. Prediction returns rows in input order and can process bounded
chunks. For very large VACs, read/write external catalog chunks and call `predict`
on each chunk; do not materialize the entire survey in memory.

Supply the same probability convention, PDZ amplitude convention, and grid as
training. zML models cannot be moved to another PDZ pipeline or grid merely
because the number of bins matches. Stored feature schemas identify required
observables. Missing/non-finite selected features raise unless the model was
explicitly trained with imputation. There is no silent row removal.

`precomputed_descriptors` is an explicit compatibility option for descriptors
computed by the same documented policy; otherwise descriptors are computed from
the supplied PDZs. Spectroscopic redshifts are not required for inference.

## Empirical-CDF numerical sensitivity

The frozen right-rank CDF is discontinuous and can contain repeated calibration
scales. In Mizuki-Wide, one repeated scale has 536 reference objects (a 1.284%
CDF jump). A CPU scale difference of about 5.3e-9 can move a query across that
step and change `r_ml` by about 0.01284. The fitted models, calibration, and
ranking definition are unchanged; bitwise percentile reproduction across
execution backends is not guaranteed. Validation checks zML/scale tolerances,
exact reference preservation, and the ranking definition, while reporting direct
percentile differences rather than silently relaxing their tolerance.

## Flags independent of the networks

```python
from turbopdz.applicability import ApplicabilityReference, compute_applicability_flags

reference = ApplicabilityReference.load("applicability.npz")
flags = compute_applicability_flags(photometry, reference)
```

The published columns are `turbopdz_in_mag_domain`,
`turbopdz_specz_support_percentile`, `turbopdz_in_specz_support_99`, and
`turbopdz_applicability_flag`. HSC uses grizy adjacent colors, median/IQR
standardization, 20-neighbor distances, and the 99% leave-one-out reference
boundary. Null-marked finite FITS sentinels are treated as missing. Invalid colors
produce NaN support percentiles and fail support/applicability flags.

Alternative ordered band sets and neighbor counts can be fitted explicitly; the
published `_99` schema keeps the 99% threshold. Low percentiles mean dense/familiar
colors, not high predicted reliability. The acceptance region follows the local
color locus; it is not one sphere centered on the sample median. The notebook's
2D PCA view cannot show the full decision boundary, and its red reference points
are sparse-color objects, not proven photo-z failures.

The compact HSC training products omit grizy. For flags, use the frozen reference
or your separately selected photometric training sample; applying flags later to
the full VAC requires the VAC photometry. No full-survey VAC is bundled here.
