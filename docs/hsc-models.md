# HSC model provenance and use

The six frozen bundles accompany Duarte & Marra, *Optimizing the extraction of
information from redshift probability distribution functions*, accepted for
publication in PASJ ([arXiv:2607.26822](https://arxiv.org/abs/2607.26822)). Each
pipeline/layer has its own native grid, preprocessing, weights, scale calibration,
and reliability CDF. All use PDZ probability masses, eleven descriptors, and
`i_cmodel_mag`. The arrays, feature order, and source checksums are in the manifests.

| Bundle | Grid bins | Grid range | PCA counts zML / sigmaML | i magnitude domain |
| --- | ---: | ---: | ---: | ---: |
| demp-wide | 601 | 0–6 | 60 / 12 | 18.5–25 |
| demp-dud | 601 | 0–6 | 6 / 12 | 19–26 |
| dnnz-wide | 100 | 0–7 | 24 / 36 | 18.5–25 |
| dnnz-dud | 100 | 0–7 | 17 / 28 | 19–26 |
| mizuki-wide | 701 | 0–7 | 22 / 45 | 18.5–25 |
| mizuki-dud | 701 | 0–7 | 31 / 46 | 19–26 |

Training uses secure HSC spectroscopy (`specz_flag_homogeneous=True`) within the
magnitude ranges above and `0.01 <= z_spec <= 5`. The flag harmonizes redshift
quality across contributing surveys; it does not make their selection function
representative of the photometric catalog. The models do not guarantee
performance outside their observed support or for another survey or PDZ pipeline.

## Fitting and hyperparameter selection

The published zML refits use 85% of the secure sample, with a 15% point holdout.
sigmaML uses five-fold OOF zML residuals on the 85% development pool, a separate
7.5% calibration subset, and a 7.5% final test. OOF excludes the object's fold
from fitting weights and preprocessing, but does not imply nested HPO.

The frozen HSC hyperparameters were selected on the original spectroscopy sample:
zML used its original 85% pool; sigmaML used a different 80/10/10 partition and
in-sample residual targets. Secure-sample refits retained those hyperparameters.
The new final-test objects were excluded from refitting and calibration, but some
entered the earlier searches. These historical models therefore do not satisfy
the strict search/test separation of a new run through the library workflow.
The [selection-pool diagnostic](../recipes/hsc-pdr3/results/hpo-selection/README.md)
reports the independent-subset checks and their limits.

New training uses one separate Optuna search for each network, with sigmaML
selection on cross-fitted targets and no calibration/test objects in either
search. Reusing the HSC hyperparameters on new partitions is a new fit; it does
not reproduce the frozen predictions bit for bit.

## Validation and interpretation

[The validation report](validation.md) compares each bundle with saved source
outputs on 256 sampled final-test objects. `sigma_ml` is an empirical error-scale
proxy, not a posterior standard deviation or a coverage guarantee. `r_ml` is a
relative score from a frozen empirical CDF; ties make it numerically sensitive
to very small scale changes. The applicability references describe photometric
support independently of the networks and do not certify redshift accuracy.

The paper's grouped permutation analysis assigns secondary-peak separation
(`distance_to_mode`) to location statistics. The library's default descriptor
blocks group it with peak descriptors. This changes grouped importance summaries,
not model inputs or predictions; use the paper's grouping for a direct comparison.

## Distribution, license, and attribution

The binaries are tracked directly in `artifacts/hsc-pdr3/` and are excluded from
Python distributions. No Git LFS, private download service, or automatic model
download is required. `SHA256SUMS` and nested manifests record the distributed
payloads. Export-time software versions remain unchanged in the frozen manifests.

The repository's [MIT license](../LICENSE) covers the authors' code and fitted
model artifacts. It does not relicense the underlying HSC data or third-party
software. Obtain source catalogs through the [HSC PDR3 archive](https://hsc-release.mtk.nao.ac.jp/doc/index.php/data-access__pdr3/)
and follow its data-use and acknowledgement requirements. Cite the turboPDZ paper
and the relevant HSC release/photo-z papers when using the models scientifically.

No raw catalog extracts, spectroscopic label vectors, or object identifiers are
bundled. Applicability files contain derived standardized color vectors and
neighbor-distance distributions, and calibration files contain derived reference
scales. The full-survey VAC is a separate planned product.
