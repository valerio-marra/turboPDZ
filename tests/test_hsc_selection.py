"""Catalog membership must survive filtering without relying on row numbers."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")
from turbopdz.recipes.hsc_selection import row_mapping


def test_membership_mapping_preserves_actual_rows_after_filtering():
    source = pd.DataFrame({"x": [1., 2., 3., 4.], "y": [np.nan, 8., 9., 10.]})
    filtered = source.iloc[[3, 0, 2]].reset_index(drop=True)
    np.testing.assert_array_equal(row_mapping(source, filtered), [3, 0, 2])


def test_membership_mapping_rejects_duplicate_or_missing_objects():
    source = pd.DataFrame({"x": [1., 2., 3.]})
    with pytest.raises(ValueError, match="unique"):
        row_mapping(pd.concat([source, source]), source)
    with pytest.raises(ValueError, match="unmatched"):
        row_mapping(source, pd.DataFrame({"x": [99.]}))
