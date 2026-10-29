# Release checklist

Use this checklist before distributing a package version or model bundle.
[The 0.1.0 validation record](release-validation.md) records completed local checks.
See [validation.md](validation.md) for test coverage and numerical-reproducibility
limits. Successful software checks do not establish full-survey performance.

- Run `pytest`, build an sdist/wheel, and test the installed wheel outside the checkout.
- Execute the complete synthetic notebook workflow through Papermill.
- Validate each converted HSC bundle against source predictions; record sample size,
  maximum differences, tested versions, and the distinction from an all-object audit.
- Review README/citation, feature/objective defaults, and artifact licensing/provenance.
- Choose binary distribution before pushing: bundles remain separate from wheels.
  This release tracks the six HSC binaries directly in `artifacts/hsc-pdr3/`,
  with `SHA256SUMS`; inference does not download them automatically.
- Archive the validated environment and declare its platform/device scope. Repeat
  the synthetic workflow on each intended GPU or production batch environment.
- Review the clean diff for credentials, private paths, catalog rows, and notebook outputs.
- Tag a version and publish only after review; no GitHub or PyPI publication is automated.
- Coordinate the full HSC VAC separately, including photometric inputs and flags.

## Reproduction scope

HSC reference bundles contain fitted weights, transforms, calibration curves, and
CDFs. Reproducing inference uses these bundles, without retraining. HSC training
configurations reuse reference hyperparameters but fit fresh models using the
library's explicit split and fixed-calibration implementation; they do not repeat
the complete reference optimization or guarantee bitwise-identical predictions.
`paper-runs.json` and bundle manifests identify the source runs. Dataset, feature,
and objective hooks define new experiments that require independent evaluation.
