# dyplom-ekg

Bachelor's thesis project: *A Unified System for Processing and Classification of
Multi-Format ECG Signals Using Machine Learning Methods*.

The system ingests ECG recordings in multiple formats (DICOM, WFDB, EDF/BDF, HL7
aECG, CSV, SCP-ECG, MIT-BIH, Philips XML, GE Muse XML), normalises them into a
unified XML representation, extracts handcrafted and learned features around
annotated heartbeats, and classifies each beat into the five AAMI categories
(N, S, V, F, Q) using a hybrid late-fusion model that combines a gradient-boosted
classifier over handcrafted features with a 1D-CNN over raw signal windows.

The unified ingestion stack lets the same trained classifier consume any
supported file format. A round-trip benchmark quantifies how classification
accuracy survives lossy format conversions.

## Layout

```
ekg/                  # new package — ML, datasets, API, CLI
├── parsers/          # unified ingestion (wraps legacy parser.py)
├── datasets/         # MIT-BIH download, AAMI mapping, DS1/DS2 splits
├── features/         # beat windows + handcrafted feature extractor
├── models/           # classical, CNN, CNN-BiLSTM, hybrid
├── training/         # train + evaluate
├── benchmark/        # multi-format robustness benchmark
├── api/              # FastAPI service
├── cli.py            # `ekg ...` entrypoint
└── legacy.py         # re-exports from main.py / parser.py

main.py, parser.py    # legacy coursework code; still the source of truth for
                      # signal processing, R-peak detection, HRV, morphology,
                      # arrhythmia detection, and format converters
tests/                # pytest suite
scripts/              # one-off helpers (dataset download, etc.)
```

## Install

```bash
pip install -e .[torch,viz,dev]
```

## Quick start

```bash
# Download MIT-BIH (~100 MB)
python -m scripts.download_mitbih

# Train classical model
ekg train --model xgboost

# Train hybrid (recommended)
ekg train --model hybrid

# Evaluate
ekg evaluate --model hybrid

# Run format robustness benchmark
ekg benchmark format-robustness

# Classify a single file (any supported format)
ekg classify path/to/record.dat --model hybrid

# Serve REST API
ekg serve --port 8000
# Then: curl -F file=@record.csv http://localhost:8000/classify
```

## CLI

| command                       | what it does                                          |
|-------------------------------|-------------------------------------------------------|
| `ekg parse <file>`            | convert any supported ECG format to unified XML       |
| `ekg classify <file>`         | run the full pipeline + ML classifier                 |
| `ekg train --model <name>`    | train one of: xgboost, rf, svm, cnn, cnn_bilstm, hybrid |
| `ekg evaluate --model <name>` | evaluate a trained checkpoint on MIT-BIH DS2          |
| `ekg benchmark format-robustness` | measure accuracy retention across format round-trips |
| `ekg serve`                   | start FastAPI service                                 |
