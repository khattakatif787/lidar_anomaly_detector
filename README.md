# LiDAR Corruption Detection Pipeline

## Overview

This project implements a complete machine learning pipeline for detecting corruptions in LiDAR point cloud data using statistical and geometric features.

The pipeline:

* extracts features from LiDAR `.pcd` files
* trains multiple classifiers
* generates detailed performance reports

---

## Dataset Structure

The pipeline expects the dataset in the following format:

```
LIHRA_DATASET/
│
├── clean_dataset/
│   ├── scenario1/
│   ├── scenario2/
│   └── ...
│
└── corrupted_dataset/
    ├── corruption_type1/
    │   ├── scenario1/
    │   └── scenario2/
    ├── corruption_type2/
    └── ...
```


## Feature Extraction

The features are specifically designed for LiDAR corruption detection.

The pipeline trains 6 classifiers.
The pipeline automatically performs:

### Step 1 — Build manifest

Creates dataset index

```
manifest.csv
```

---

### Step 2 — Train/Test split
Group-based split to prevent leakage

```
manifest_split.csv
```

---

### Step 3 — Feature extraction

Saves:

```
feat_cache/
    X_train.npy
    X_test.npy
    y_train.npy
    y_test.npy
```

---

### Step 4 — Train classifiers

Saves:

```
models/
    lr/
    rf/
    xgboost/
    ...
```

Each model saves:

```
model.joblib
test_predictions.csv
```

---

### Step 5 — Generate final report

Creates:

```
summary_overall.csv
summary_fpr_at_alr.csv
```

Containing:

* TP
* TN
* FP
* FN
and FPRs for specific FNRs
---

## How to Install

Clone repository:

```
git clone 
cd lidar_anomaly_detector
```

Create environment:

```
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## How to Run

Run full experiment:

```
chmod +x run_all_cpc.sh
./run_all_cpc.sh
```

---

## Output

Results are saved in:

```
runs/
```

Example:

```
runs/CPC3_f_0.01/
```

---

## Reproducibility

The pipeline is fully reproducible.

Uses:

* fixed random seed
* deterministic train/test split

---

## Project Structure

```
lidar_anomaly_detector/

run_full_pipeline.py
compute_fpr_at_alr.py
run_all_cpc.sh
requirements.txt

src/lidar_anom/

    features_basic.py
    data/
        scan.py
        split.py
    eval_report.py
```

---

## Purpose

This project supports research on:

Safe LiDAR perception

Corruption detection

---

## License

Academic use only
