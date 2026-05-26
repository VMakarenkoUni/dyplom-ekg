"""Dataset module tests that don't require the MIT-BIH download."""

from __future__ import annotations

from ekg.datasets.mitbih import aami_label
from ekg.datasets.splits import (
    DS1_RECORDS,
    DS2_RECORDS,
    EXCLUDED_RECORDS,
    ds1_ds2_split,
)


def test_aami_mapping_known_symbols():
    assert aami_label("N") == "N"
    assert aami_label("L") == "N"
    assert aami_label("R") == "N"
    assert aami_label("A") == "S"
    assert aami_label("V") == "V"
    assert aami_label("E") == "V"
    assert aami_label("F") == "F"
    assert aami_label("/") == "Q"
    assert aami_label("zzz") is None


def test_ds1_ds2_disjoint_and_complete():
    ds1, ds2 = ds1_ds2_split()
    assert ds1 == DS1_RECORDS
    assert ds2 == DS2_RECORDS
    assert set(ds1).isdisjoint(set(ds2))
    overlap = set(ds1) | set(ds2)
    assert overlap.isdisjoint(EXCLUDED_RECORDS)
    # de Chazal split has 22 + 22 = 44 records (4 paced excluded).
    assert len(ds1) == 22
    assert len(ds2) == 22
