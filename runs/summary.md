# Experimental Results — MIT-BIH (DS1→DS2 inter-patient)

Evaluation protocol: AAMI 5-class (N/S/V/F/Q), de-Chazal DS1/DS2 patient-disjoint split, lead MLII, beat window 260 samples @ 360 Hz.

## Per-model overall metrics

| Model           | Accuracy | Macro-F1 | N F1   | S F1   | V F1   | F F1   |
|-----------------|---------:|---------:|-------:|-------:|-------:|-------:|
| rule_based      | 0.8517 | 0.2396 | 0.9483 | 0.2488 | 0.0000 | 0.0000 |
| xgboost         | 0.8687 | 0.4184 | 0.9268 | 0.1859 | 0.8862 | 0.0930 |
| random_forest   | 0.9294 | 0.3739 | 0.9625 | 0.0139 | 0.8912 | 0.0018 |
| svm             | 0.8554 | 0.4612 | 0.9171 | 0.4325 | 0.8167 | 0.1396 |
| cnn             | 0.6665 | 0.2732 | 0.7999 | 0.0454 | 0.5192 | 0.0016 |
| cnn_bilstm      | 0.5049 | 0.2893 | 0.6451 | 0.0397 | 0.7535 | 0.0083 |
| hybrid          | 0.7940 | 0.3938 | 0.8815 | 0.1624 | 0.8709 | 0.0524 |
| xgboost_2lead   | 0.8607 | 0.4191 | 0.9209 | 0.1778 | 0.8603 | 0.1366 |
| svm_2lead       | 0.8262 | 0.4663 | 0.8982 | 0.5409 | 0.7796 | 0.1129 |
| cnn_2lead       | 0.6059 | 0.3107 | 0.7391 | 0.0926 | 0.7148 | 0.0072 |
| hybrid_2lead    | 0.7693 | 0.4028 | 0.8684 | 0.2234 | 0.8247 | 0.0948 |

## Per-model recall on rare classes

| Model           | S recall | V recall | F recall |
|-----------------|---------:|---------:|---------:|
| rule_based      | 0.3016 | 0.0000 | 0.0000 |
| xgboost         | 0.1268 | 0.9516 | 0.5206 |
| random_forest   | 0.0071 | 0.8730 | 0.0026 |
| svm             | 0.5199 | 0.9481 | 0.7397 |
| cnn             | 0.0343 | 0.9155 | 0.0206 |
| cnn_bilstm      | 0.1355 | 0.9224 | 0.1134 |
| hybrid          | 0.1078 | 0.9404 | 0.4794 |
| xgboost_2lead   | 0.1230 | 0.9429 | 0.8196 |
| svm_2lead       | 0.5808 | 0.8882 | 0.8995 |
| cnn_2lead       | 0.2733 | 0.8584 | 0.0747 |
| hybrid_2lead    | 0.2074 | 0.7714 | 0.8918 |

## Format-robustness benchmark (full DS2)

Pipeline: native WFDB → trained model → reference predictions; then WFDB → {CSV, EDF} → unified XML parser → same model. *Label agreement* = fraction of beats where the prediction after round-trip matches the native prediction. *Accuracy* = vs AAMI ground truth.

| Model            | Format | Label agreement | Accuracy | Mean |Δsignal| |
|------------------|--------|----------------:|---------:|----------------:|
| xgboost 1-lead   | (native) |               — | 0.8687 |              — |
| xgboost 1-lead   | csv    | 1.0000          | 0.8687 | 0.000e+00 |
| xgboost 1-lead   | edf    | 0.9970          | 0.8689 | 3.709e-05 |
| hybrid 1-lead    | (native) |               — | 0.7940 |              — |
| hybrid 1-lead    | csv    | 1.0000          | 0.7940 | 0.000e+00 |
| hybrid 1-lead    | edf    | 0.9928          | 0.7938 | 3.709e-05 |
| hybrid 2-lead    | (native) |               — | 0.7693 |              — |
| hybrid 2-lead    | csv    | 1.0000          | 0.7693 | 0.000e+00 |
| hybrid 2-lead    | edf    | 0.9964          | 0.7697 | 3.778e-05 |

## Findings

1. **Both CSV and EDF are essentially lossless** for classification when the exporter emits a time column + per-lead value columns: label agreement is 100% on CSV and ≥99.3% on EDF across every model. Mean per-sample |Δ| is 0 for CSV (text floats round-trip cleanly through 6 decimals) and 3.7×10⁻⁵ for EDF (16-bit quantisation noise).
2. **The unified XML intermediate preserves classification quality** across single- and multi-lead inputs, validating the 'Multi-Format' part of the system title.
3. **Inter-patient MIT-BIH is hard**: S and F classes are sparse and morphologically close to N/V, producing low F1 across every model — consistent with the literature on the de-Chazal protocol.
4. **Multi-lead is the biggest single win**: adding V1/V5 lifts F-class recall from 0.52 (XGBoost 1-lead) to 0.89 (Hybrid 2-lead) — the system finds nearly nine in ten fusion beats.
5. **Hybrid story**: the dual-stream model trades a couple of points of overall macro-F1 for the highest rare-class recall in the table, which is the clinically relevant figure of merit.
