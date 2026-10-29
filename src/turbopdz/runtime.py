"""Narrow handling of optional, cosmetic notebook integration warnings."""

from __future__ import annotations

import warnings


def import_optuna():
    """Import Optuna without tqdm's optional-widget warning.

    The workflow does not request notebook progress widgets. All other warnings,
    including numerical, experimental, resource and security warnings, remain
    visible. This does not modify process-wide warning filters.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"IProgress not found.*",
                                module=r"tqdm\.auto")
        import optuna
    return optuna
