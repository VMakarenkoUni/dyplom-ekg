"""Inter-patient DS1/DS2 split for MIT-BIH (de Chazal et al., 2004).

This split is the de-facto standard for credible MIT-BIH benchmarks:
training and test records are patient-disjoint, which prevents the
optimistic-bias of intra-patient random splits.

Reference:
    de Chazal P., O'Dwyer M., Reilly R. B. "Automatic classification of
    heartbeats using ECG morphology and heartbeat interval features."
    IEEE Trans. Biomed. Eng., 2004.
"""

from __future__ import annotations

from typing import Iterable

DS1_RECORDS: tuple[str, ...] = (
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
    "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
    "223", "230",
)

DS2_RECORDS: tuple[str, ...] = (
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
    "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
    "233", "234",
)

# Records 102, 104, 107, 217 are excluded by the original protocol — they
# contain paced beats that the AAMI standard tells us to remove from
# evaluation of beat classifiers.
EXCLUDED_RECORDS: frozenset[str] = frozenset({"102", "104", "107", "217"})


def ds1_ds2_split() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the (training, test) record-id tuples."""
    return DS1_RECORDS, DS2_RECORDS


def is_excluded(record_id: str) -> bool:
    return record_id in EXCLUDED_RECORDS


def all_records() -> Iterable[str]:
    yield from DS1_RECORDS
    yield from DS2_RECORDS
