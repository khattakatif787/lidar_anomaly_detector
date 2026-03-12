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
###**Corrupted Dataset Generation**
LiHRA dataset contains only clean LiDAR pint clouds frames, corrupted versions of the point clouds frames were generated.  Each corruption type was applied using multiple parameter values to simulate different fault severity levels and operating conditions using GitHub repository (https://github.com/GKnerd/lidar_fault_injection). The following corruption were injected:


**1.	Sparse points (CPC3)**
The following corruptions are injected:
•	Random Dropout: where a percentage of points are randomly removed. The dropout rates ranging from 0.1 to 0.9 are used.
•	Structured Dropout: where points were removed according to specific spatial patterns, including sector-based, distance-based, and checkerboard patterns, while the intensity of dropout was 0.5.
•	Sparse Scan: where point cloud sparsity was increased using sparsity factors ranging from 2 to 10.

**2.	Point dropouts and occlusion (CPC4)**
This category simulates obstruction of the LiDAR sensor or reduction in its field of view. The following corruptions are injected:
•	Occlusion: where portions of the point cloud were removed using artificial occlusion spheres. The occlusion patch size was set to 1, placed at a distance of 50, and the number of occlusion spheres was varied from 1 to 5 to simulate different levels of obstruction.
•	Field-of-View Reduction: where the angular coverage of the LiDAR scan was reduced by cropping 50% of the azimuth (horizontal), 50% of the elevation (vertical), and 50% of both azimuth and elevation to simulate limited sensor coverage.
•	Random Dropout: where a percentage of points are randomly removed. The dropout rates ranging from 0.1 to 0.9 are used.
•	Structured Dropout: where points were removed according to specific spatial patterns, including sector-based, distance-based, and checkerboard patterns, while the intensity of dropout was 0.5.

**3.	Measurement noise effecting point coordinates (CPC6)**
This category simulates sensor measurement errors and the following noise corruptions are injected:
•	Crosstalk Noise: with severity levels from 1 to 5.
•	Impulse Noise: with severity levels from 1 to 5.
•	Gaussian Noise: where noise was added to point coordinates using a standard deviation of position jitter of 0.01 m and 0.02 m, an outlier fraction of 0.03 and 0.05, and an outlier noise sigma of 0.2 to simulate measurement uncertainty and positional distortion.


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
