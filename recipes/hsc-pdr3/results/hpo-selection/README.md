# HPO selection-pool diagnostic

The CSVs contain aggregate metrics for the six frozen HSC configurations in
`paper-runs.json`. `provenance.json` records run identities, data checksums,
cohort counts, and the exact matching rule. No individual catalog rows or
identifiers are included.

Cohorts are the full current test sample, objects excluded from the original
zML search pool (`zml_hpo_unseen`), objects excluded from the original sigmaML
search pool (`sigma_hpo_unseen`), objects excluded from both (`joint_hpo_unseen`),
and the remaining reliability-test objects (`any_hpo_exposed`). Rows are matched
using 12 unchanged numeric columns; every unique hash match is verified against
all column values. Filtered row numbers are not treated as persistent IDs.

Recompute with the source-run workspace and its comparison tables:

```bash
python -m turbopdz.recipes.hsc_selection \
  --project-root /path/to/reference-workspace \
  --out /path/to/new-hpo-check --n-boot 2000
```

Use `--n-boot 0` to recompute observed metrics and provenance only. This requires
the original and secure-spectroscopy runs and compact data products, which are
not distributed. No fitting is performed.

Positive improvements favor turboPDZ. The jointly excluded bootstrap resamples
within ten full-test redshift strata, keeping stratum counts fixed and sharing
object draws across each paired model comparison. Equal-sized exposed controls
are resampled in the same strata. The CSV also retains these secondary gap
comparisons; they are not a test of numerical equivalence or HPO optimality.
Intervals condition on the saved fitted models and calibration.

The native scatter-filtering improvement persists in all six jointly excluded
cohorts; its two pairwise intervals are positive in five configurations and
unresolved in DNNz-DUD. Other metrics and secondary comparisons need not preserve
the same ordering. See the [model provenance](../../../../docs/hsc-models.md).
