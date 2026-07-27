# turboPDZ

**Optimizing the extraction of information from redshift probability distribution functions.**

turboPDZ is a machine-learning framework that produces optimized photometric-redshift
point estimates ($z_{\rm ml}$) and data-driven reliability scores ($r_{\rm ml}$) directly
from redshift probability distribution functions (PDZs). The pipeline is survey-independent
and compatible with any photo-$z$ method that delivers per-galaxy PDZs.

## Method

Each PDZ is compressed via PCA and combined with summary descriptors (mean, median, mode,
width, skewness, kurtosis, and peak statistics). A multilayer perceptron, optimized with
[Optuna](https://optuna.org/) under a composite objective (σ_NMAD and outlier fraction),
produces $z_{\rm ml}$. A second network, trained in log-space on the normalized redshift
error, yields a calibrated uncertainty $\sigma_{\rm ml}$, from which the reliability score
$r_{\rm ml}$ is derived via percentile ranking.

The framework is demonstrated on the three independent photo-$z$ pipelines (DEmP, DNNz,
Mizuki) of the HSC-SSP PDR3, across both Wide and Deep/UltraDeep layers. $z_{\rm ml}$
outperforms the catalog best estimate in scatter, bias, and outlier fraction across all
six pipeline–layer combinations, while $r_{\rm ml}$ provides more efficient galaxy
filtering than standard catalog quality indicators.

## Reference

Duarte & Marra, *Optimizing the extraction of information from redshift probability
distribution functions*, PASJ (submitted).

## Availability

The source code will be made publicly available upon publication of the paper.
Trained models and the optimized quantities ($z_{\rm ml}$, $\sigma_{\rm ml}$, $r_{\rm ml}$)
for all HSC-SSP PDR3 objects will be released as a value-added catalog.
