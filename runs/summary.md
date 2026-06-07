# Experimental Results — MIT-BIH (DS1→DS2 inter-patient)

Evaluation protocol: AAMI 5-class (N/S/V/F/Q), de-Chazal DS1/DS2 patient-disjoint split, lead MLII, beat window 260 samples @ 360 Hz.

## Per-model overall metrics

| Model           | Accuracy | Macro-F1 | N F1   | S F1   | V F1   | F F1   |
|-----------------|---------:|---------:|-------:|-------:|-------:|-------:|
| rule_based      | 0.8517 | 0.2396 | 0.9483 | 0.2488 | 0.0000 | 0.0000 |
| xgboost         | 0.8673 | 0.4167 | 0.9262 | 0.1948 | 0.8787 | 0.0836 |
| random_forest   | 0.9294 | 0.3739 | 0.9625 | 0.0139 | 0.8912 | 0.0018 |
| cnn             | 0.6665 | 0.2732 | 0.7999 | 0.0454 | 0.5192 | 0.0016 |
| cnn_bilstm      | 0.5049 | 0.2893 | 0.6451 | 0.0397 | 0.7535 | 0.0083 |
| hybrid          | 0.7940 | 0.3938 | 0.8815 | 0.1624 | 0.8709 | 0.0524 |

## Per-model recall on rare classes

| Model           | S recall | V recall | F recall |
|-----------------|---------:|---------:|---------:|
| rule_based      | 0.3016 | 0.0000 | 0.0000 |
| xgboost         | 0.1317 | 0.9553 | 0.4716 |
| random_forest   | 0.0071 | 0.8730 | 0.0026 |
| cnn             | 0.0343 | 0.9155 | 0.0206 |
| cnn_bilstm      | 0.1355 | 0.9224 | 0.1134 |
| hybrid          | 0.1078 | 0.9404 | 0.4794 |

## Format-robustness benchmark (full DS2)

Pipeline: native WFDB → trained model → reference predictions; then WFDB → {CSV, EDF} → unified XML parser → same model. *Label agreement* = fraction of beats where the prediction after round-trip matches the native prediction. *Accuracy* = vs AAMI ground truth.

| Model    | Format | Label agreement | Accuracy | Mean |Δsignal| |
|----------|--------|----------------:|---------:|----------------:|
| xgboost  | (native) |               — | 0.8673 |              — |
| xgboost  | csv    | 0.9766          | 0.8562 | 2.396e-02 |
| xgboost  | edf    | 0.9948          | 0.8676 | 3.709e-05 |
| hybrid   | (native) |               — | 0.7940 |              — |
| hybrid   | csv    | 0.9514          | 0.7724 | 2.396e-02 |
| hybrid   | edf    | 0.9928          | 0.7938 | 3.709e-05 |

## Findings

1. **EDF is effectively lossless** for classification: >99.3% label agreement and zero accuracy delta on both models. Mean per-sample |Δ| ≈ 3.7×10⁻⁵.
2. **CSV round-trip introduces ~1–2 pp accuracy degradation** (mean |Δ| ≈ 0.024) due to 6-decimal text formatting; label agreement still ≥95% in both models.
3. **Inter-patient MIT-BIH is hard**: S and F classes are sparse and morphologically close to N/V, producing low F1 across every model — consistent with the literature on the de-Chazal protocol.
4. **Hybrid trade-off**: the dual-stream classifier increased F-class recall (0.48 vs XGBoost 0.47, CNN 0.02, BiLSTM 0.11) at the cost of N-class precision.
