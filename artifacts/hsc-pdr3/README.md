# Frozen HSC-PDR3 model bundles

Six directories cover DEmP, DNNz, and Mizuki in Wide and Deep/UltraDeep (`dud`).
Each contains `zml/`, `sigmaml/`, and a layer-specific `applicability.npz` when
available. Nested manifests record hashes, feature order/policy, native grids,
source run tags, selected parameters, software versions, and the parent-model link.

The bundles contain no raw catalog tables, spectroscopic label vectors, or object
identifiers. Applicability references contain derived standardized color vectors
needed for neighbor queries, with their reference-selection provenance. They
cannot establish that the spectroscopy is representative of the full survey.

`validation.json` records sampled comparisons with source predictions. TensorFlow
is required to load the models; Optuna is not. Model bundles are intentionally not
included in the Python wheel. They are tracked directly in Git, so a normal clone
includes all six pairs without Git LFS or an automatic download. `SHA256SUMS`
records every distributed bundle file; nested manifests are verified on load.
See [model provenance and license scope](../../docs/hsc-models.md), the
[release checklist](../../docs/release-checklist.md), and
[source run identities](../../recipes/hsc-pdr3/paper-runs.json).
Never replace a released bundle in place; version a changed one.
