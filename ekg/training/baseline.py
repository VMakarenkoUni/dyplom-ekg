"""Rule-based baseline derived from the legacy `ArrhythmiaDetector`.

This is intentionally simple — it turns the legacy rule outputs into a
5-class prediction so the trained ML models can be compared against
the existing coursework system on the same DS2 records and metrics.

Mapping rule:
  - bradycardia / tachycardia / pause → still "N" (those are rhythm-level
    flags, not per-beat morphological classes; for a beat-level baseline
    we use the per-beat heuristic);
  - PVC / wide QRS at this beat        → "V";
  - PAC / supraventricular at this beat → "S";
  - fusion beat detected               → "F";
  - paced / artifact                   → "Q";
  - otherwise                          → "N".

The mapping is intentionally conservative; the point of the baseline is to
demonstrate the gap between rules and ML rather than to win the comparison.
"""

from __future__ import annotations

import numpy as np

from ekg import AAMI_CLASSES

LABEL_TO_INT = {c: i for i, c in enumerate(AAMI_CLASSES)}


def predict_baseline(bundle: dict) -> np.ndarray:
    """Per-beat rule-based prediction over a prepared MIT-BIH bundle.

    Uses RR-interval features (rr_pre, rr_post) and morphology features
    (qrs_width, r_amp) already present in `bundle['features']` to mirror
    the heuristics in the legacy `ArrhythmiaDetector` without re-running it.
    """
    features = bundle["features"]
    n = features.shape[0]
    if n == 0:
        return np.empty((0,), dtype=np.int64)

    # Indices align with FEATURE_NAMES in ekg.features.handcrafted.
    rr_pre = features[:, 0]
    rr_post = features[:, 1]
    rr_ratio = features[:, 2]
    qrs_width = features[:, 5]
    r_amp = features[:, 6]

    preds = np.full(n, LABEL_TO_INT["N"], dtype=np.int64)

    wide_qrs = qrs_width > 0.12      # >120 ms QRS → ventricular ectopic
    preds[wide_qrs] = LABEL_TO_INT["V"]

    very_low_amp = r_amp < 0.15
    preds[very_low_amp & ~wide_qrs] = LABEL_TO_INT["Q"]

    premature = (rr_pre > 0) & (rr_pre < 0.7 * np.where(rr_post > 0, rr_post, rr_pre))
    sv_ectopic = premature & ~wide_qrs & ~very_low_amp
    preds[sv_ectopic] = LABEL_TO_INT["S"]

    fusion = wide_qrs & (rr_ratio > 0.85) & (rr_ratio < 1.15)
    preds[fusion] = LABEL_TO_INT["F"]

    return preds
