"""Check sensitivity to membership in the original HPO pools.

Run from the repository's Python environment, for example::

    python -m turbopdz.recipes.hsc_selection --project-root /path/to/reference-workspace \
        --out /path/to/new-hpo-check --n-boot 2000

Only saved predictions, split indices, and catalog columns are read. No model
or preprocessing object is fitted. Aggregate CSVs and provenance are written
only to --out; optional bootstrap draws require --save-draws. Positive reported
improvements favor turboPDZ. This conditional diagnostic is not an equivalence
test or a new hyperparameter search.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .hsc_bootstrap import (
    auc_metric_pairs,
    correlation_metric_pairs,
    load_configuration,
    point_metric_pairs,
    validate_comparison_products,
)

KEY_COLUMNS = [
    'i_cmodel_mag', 'specz_redshift', 'photoz_mean', 'photoz_median',
    'photoz_mode', 'photoz_best', 'photoz_std_best', 'photoz_std_mean',
    'photoz_err68_min', 'photoz_err68_max', 'photoz_risk_best', 'photoz_conf_best',
]
CASES = [
    'demp_dud', 'demp_wide', 'dnnz_dud',
    'dnnz_wide', 'mizuki_dud', 'mizuki_wide',
]


def json_read(path: Path) -> dict:
    """Read saved run metadata."""
    return json.loads(path.read_text())


def read_rows(path: Path) -> pd.DataFrame:
    """Read only the unchanged numeric columns needed for object matching."""
    return pq.read_table(path, columns=KEY_COLUMNS).to_pandas().astype('float64')


def row_mapping(source: pd.DataFrame, current: pd.DataFrame) -> np.ndarray:
    """Match catalog rows and reject ambiguous or incomplete correspondence.

    Parameters
    ----------
    source, current
        Numeric catalog frames containing the same unchanged columns.

    Returns
    -------
    numpy.ndarray
        Original row index for each current object. Hash matches are checked
        against all column values before the mapping is accepted.
    """
    source_keys = pd.util.hash_pandas_object(source, index=False)
    current_keys = pd.util.hash_pandas_object(current, index=False)
    if not (source_keys.is_unique and current_keys.is_unique):
        raise ValueError('Object matching requires unique catalog keys')
    mapping = pd.Index(source_keys).get_indexer(current_keys)
    if (mapping < 0).any():
        raise ValueError('Current catalog contains unmatched objects')
    if not np.array_equal(
        source.iloc[mapping].to_numpy(), current.to_numpy(), equal_nan=True
    ):
        raise ValueError('Hash matches failed exact column-value validation')
    return mapping


def bias_comparison(
    original_directory: Path,
    point: dict[str, np.ndarray],
    original: pd.DataFrame,
    original_rows: np.ndarray,
    current_test_index: np.ndarray,
) -> dict:
    """Compare original and revised biases without loading or refitting models.

    The secure subset is identified by membership in the current catalog. The
    shared cohort is the intersection of the two saved point-test partitions.
    """
    with np.load(original_directory / 'split_idx.npz', allow_pickle=False) as data:
        original_test = data['test_idx'].copy()
    with np.load(
        original_directory / 'test_outputs.npz', allow_pickle=False
    ) as data:
        z_spec = data['y_test'].astype(float)
        z_pred = data['z_pred_test'].astype(float)
    np.testing.assert_allclose(
        z_spec, original['specz_redshift'].to_numpy()[original_test],
        rtol=1e-6, atol=1e-6,
    )
    residual = (z_pred - z_spec) / (1 + z_spec)
    secure = np.isin(original_test, original_rows)
    old_position = pd.Index(original_test).get_indexer(
        original_rows[current_test_index]
    )
    shared = old_position >= 0
    np.testing.assert_allclose(
        z_spec[old_position[shared]], point['z_spec'][shared],
        rtol=1e-6, atol=1e-6,
    )
    return {
        'original_full_test_n': len(original_test),
        'original_full_test_bias': float(np.median(residual)),
        'original_secure_test_n': int(secure.sum()),
        'original_secure_test_bias': float(np.median(residual[secure])),
        'shared_test_n': int(shared.sum()),
        'original_shared_test_bias': float(
            np.median(residual[old_position[shared]])
        ),
        'revised_shared_test_bias': float(np.median(point['dz_ml'][shared])),
        'revised_full_test_bias': float(np.median(point['dz_ml'])),
    }


def load_case(project_root: str | Path, case: str) -> tuple:
    """Load frozen outputs and reconstruct original search-pool membership."""
    root = Path(project_root)
    models = root / 'models_optuna'
    sigma_dir, = models.glob(f'sigmaml_v2-{case}-*s50_homospecz*')
    sigma_info = json_read(sigma_dir / 'run_info.json')
    zml_dir = models / sigma_info['run_tag_zml']
    zml_info = json_read(zml_dir / 'run_info.json')
    old_zml_dir = models / zml_info['run_tag_zml_source']
    old_sigma_dir = models / sigma_info['run_tag_sigmaml_source']
    old_sigma_info = json_read(old_sigma_dir / 'run_info.json')
    current = read_rows(zml_dir / 'table_pd_zml.parquet')
    original = read_rows(old_zml_dir / 'table_pd_zml.parquet')
    mapping = row_mapping(original, current)
    sigma_source_path = root / old_sigma_info['data_files']['zml_table_parquet']
    sigma_mapping = (
        mapping if sigma_source_path == old_zml_dir / 'table_pd_zml.parquet'
        else row_mapping(read_rows(sigma_source_path), current)
    )
    with np.load(old_zml_dir / 'split_idx.npz', allow_pickle=False) as values:
        original_zml_pool = values['pool_idx'].copy()
    with np.load(old_sigma_dir / 'split_idx.npz', allow_pickle=False) as values:
        original_sigma_pool = values['trainval_idx'].copy()
    with np.load(zml_dir / 'split_idx.npz', allow_pickle=False) as values:
        point_index = values['test_idx'].copy()
    with np.load(sigma_dir / 'split_idx.npz', allow_pickle=False) as values:
        final_index = values['test_idx'].copy()
    spec = dict(
        cat=sigma_info['cat'], layer=sigma_info['layer'], seed=50,
        zml_run=zml_dir.name, sigmaml_run=sigma_dir.name,
    )
    point, reliability, provenance = load_configuration(root, spec)
    assert len(point_index) == len(point['dz_ml'])
    assert len(final_index) == len(reliability['dz_ml'])
    z_spec = current['specz_redshift'].to_numpy()[final_index]
    unseen_zml = ~np.isin(mapping, original_zml_pool)
    unseen_sigma = ~np.isin(sigma_mapping, original_sigma_pool)
    clean_point = unseen_zml[point_index]
    clean = (unseen_zml & unseen_sigma)[final_index]
    source_sigma_unseen = unseen_sigma[final_index]
    provenance.update(
        case=case, original_zml_search=zml_info['run_tag_zml_source'],
        original_sigma_search=sigma_info['run_tag_sigmaml_source'],
        n_hpo_unseen_point=int(clean_point.sum()),
        n_joint_hpo_unseen_reliability=int(clean.sum()),
        n_sigma_hpo_unseen_reliability=int(source_sigma_unseen.sum()),
        matching=(
            'Unique exact equality of 12 unchanged columns; '
            'hash keys verified against all values'
        ),
    )
    if case == 'mizuki_dud':
        provenance['bias_diagnostic'] = bias_comparison(
            old_zml_dir, point, original, mapping, point_index
        )
    return (
        point, reliability, z_spec, clean_point, clean,
        source_sigma_unseen, provenance, spec,
    )


def take(data: dict[str, np.ndarray], index: np.ndarray) -> dict[str, np.ndarray]:
    """Apply one object mask or index vector to every paired prediction."""
    return {name: values[index] for name, values in data.items()}


def metric_rows(
    case: str, cohort: str, family: str, pairs: dict, n: int
) -> list[dict]:
    """Build aggregate rows with a positive-is-better improvement convention."""
    rows = []
    for key, (candidate, reference) in pairs.items():
        improvement = (
            candidate - reference if family == 'correlation'
            else reference - candidate
        )
        rows.append(dict(
            case=case, cohort=cohort, family=family, key=key,
            n=n, candidate=candidate, reference=reference,
            improvement=improvement,
        ))
    return rows


def improvements(data: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute all paired correlation and filtering-AUC improvements."""
    corr = correlation_metric_pairs(data)
    auc = auc_metric_pairs(data)
    return {
        **{
            'correlation::' + key: candidate - reference
            for key, (candidate, reference) in corr.items()
        },
        **{
            'auc::' + key: reference - candidate
            for key, (candidate, reference) in auc.items()
        },
    }


def bootstrap_exposure(
    data: dict[str, np.ndarray],
    z_spec: np.ndarray,
    clean: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Match exposed controls to clean z bins/counts; paired models per draw.

    Both groups are resampled at the clean group's size. This avoids mistaking
    finite-size AUC/quantile behavior for exposure sensitivity. Stratum counts
    and the calibration/CDF stay fixed; these are conditional diagnostics, not
    a reoptimization experiment or proof of numerical equivalence.

    Parameters
    ----------
    data
        Paired frozen predictions and residuals on the final-test sample.
    z_spec, clean
        Spectroscopic redshifts and the mask excluding both old search pools.
    n_boot, seed
        Number of replicates and the random-generator seed.

    Returns
    -------
    dict
        Each metric maps to an array with excluded-group improvement,
        matched-control improvement, and their difference in its columns.
    """
    edges = np.unique(np.quantile(z_spec, np.linspace(0, 1, 11)))
    bins = np.searchsorted(edges[1:-1], z_spec, side='right')
    clean_blocks = [
        np.flatnonzero(clean & (bins == value)) for value in np.unique(bins)
    ]
    exposed_blocks = [
        np.flatnonzero(~clean & (bins == value)) for value in np.unique(bins)
    ]
    if not all(len(e) or not len(c)
               for c, e in zip(clean_blocks, exposed_blocks)):
        raise ValueError('A redshift stratum has no exposed control objects')
    keys = list(improvements(take(data, clean)))
    draws = {key: np.empty((n_boot, 3)) for key in keys}
    rng = np.random.default_rng(seed)
    started = time.monotonic()
    for rep in range(n_boot):
        clean_index = np.concatenate([
            rng.choice(block, size=len(block), replace=True)
            for block in clean_blocks if len(block)
        ])
        exposed_index = np.concatenate([
            rng.choice(block, size=len(c), replace=True)
            for c, block in zip(clean_blocks, exposed_blocks) if len(c)
        ])
        clean_metrics = improvements(take(data, clean_index))
        exposed_metrics = improvements(take(data, exposed_index))
        for key in keys:
            draws[key][rep] = (
                clean_metrics[key], exposed_metrics[key],
                clean_metrics[key] - exposed_metrics[key],
            )
        if (rep + 1) % 200 == 0:
            elapsed = time.monotonic() - started
            print(
                f'  {rep + 1}/{n_boot} matched bootstrap pairs ({elapsed:.1f}s)',
                flush=True,
            )
    return draws


def main(argv: list[str] | None = None) -> None:
    """Validate baseline tables and write independent-subset diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--n-boot', type=int, default=0)
    parser.add_argument('--save-draws', action='store_true')
    args = parser.parse_args(argv)
    if args.n_boot < 0:
        parser.error('--n-boot must be non-negative')
    args.out.mkdir(parents=True, exist_ok=True)
    observed_rows, summaries, provenance_rows = [], [], []
    for case in CASES:
        (
            point, data, z, clean_point, clean, sigma_unseen, provenance, spec
        ) = load_case(args.project_root, case)
        provenance_rows.append(provenance)
        full_point = point_metric_pairs(point)
        full_corr = correlation_metric_pairs(data)
        full_auc = auc_metric_pairs(data)
        validate_comparison_products(
            args.project_root, spec, point_pairs=full_point,
            correlation_pairs=full_corr, auc_pairs=full_auc,
        )
        for cohort, mask in [
            ('full', np.ones(len(point['dz_ml']), bool)),
            ('zml_hpo_unseen', clean_point),
        ]:
            observed_rows.extend(metric_rows(
                case, cohort, 'point', point_metric_pairs(take(point, mask)),
                int(mask.sum()),
            ))
        for cohort, mask in [
            ('full', np.ones(len(clean), bool)), ('joint_hpo_unseen', clean),
            ('sigma_hpo_unseen', sigma_unseen), ('any_hpo_exposed', ~clean),
        ]:
            subset = take(data, mask)
            observed_rows.extend(metric_rows(
                case, cohort, 'correlation', correlation_metric_pairs(subset),
                int(mask.sum()),
            ))
            observed_rows.extend(metric_rows(
                case, cohort, 'auc', auc_metric_pairs(subset), int(mask.sum()),
            ))
        pd.DataFrame(observed_rows).to_csv(
            args.out / 'observed_metrics.csv', index=False
        )
        (args.out / 'provenance.json').write_text(
            json.dumps(provenance_rows, indent=2)
        )
        print(
            case, 'full', len(clean), 'sigma-unseen', int(sigma_unseen.sum()),
            'joint-unseen', int(clean.sum()), 'baseline tables verified',
            flush=True,
        )
        if args.n_boot:
            draws = bootstrap_exposure(
                data, z, clean, args.n_boot, 20261027 + CASES.index(case)
            )
            if args.save_draws:
                np.savez_compressed(
                    args.out / f'{case}_matched_draws.npz', **draws
                )
            observed_clean = improvements(take(data, clean))
            for key, values in draws.items():
                clean_low, clean_high = np.quantile(values[:, 0], [.025, .975])
                gap_low, gap_high = np.quantile(values[:, 2], [.025, .975])
                summaries.append(dict(
                    case=case, key=key, n_clean=int(clean.sum()),
                    clean_improvement=float(observed_clean[key]),
                    clean_ci_low=clean_low, clean_ci_high=clean_high,
                    matched_exposed_median=float(np.median(values[:, 1])),
                    gap_median=float(np.median(values[:, 2])), gap_ci_low=gap_low,
                    gap_ci_high=gap_high, n_boot=args.n_boot,
                ))
            pd.DataFrame(summaries).to_csv(
                args.out / 'bootstrap_summary.csv', index=False
            )


if __name__ == '__main__':
    main()
