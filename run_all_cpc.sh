#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="/home/users/muhammad.atif/LIHRA_DATASET"

CPC3_CORRUPTIONS="random_dropout_rate0.1 random_dropout_rate0.2 random_dropout_rate0.3 random_dropout_rate0.4 random_dropout_rate0.5 random_dropout_rate0.6 random_dropout_rate0.7 random_dropout_rate0.8 random_dropout_rate0.9 structured_dropout_rate0.5_patterncheckerboard structured_dropout_rate0.5_patterndistance structured_dropout_rate0.5_patternsector sparse_scan_pattern_sparsity2.0 sparse_scan_pattern_sparsity3.0 sparse_scan_pattern_sparsity4.0 sparse_scan_pattern_sparsity5.0 sparse_scan_pattern_sparsity6.0 sparse_scan_pattern_sparsity8.0 sparse_scan_pattern_sparsity10.0"

CPC4_CORRUPTIONS="random_dropout_rate0.1 random_dropout_rate0.2 random_dropout_rate0.3 random_dropout_rate0.4 random_dropout_rate0.5 random_dropout_rate0.6 random_dropout_rate0.7 random_dropout_rate0.8 random_dropout_rate0.9 structured_dropout_rate0.5_patterncheckerboard structured_dropout_rate0.5_patterndistance structured_dropout_rate0.5_patternsector simulate_occlusion_dist50_count1_size1.0 simulate_occlusion_dist50_count2_size1.0 simulate_occlusion_dist50_count3_size1.0 simulate_occlusion_dist50_count4_size1.0 simulate_occlusion_dist50_count5_size1.0 reduce_fov_h0.5_v0.0 reduce_fov_vertical_0.5 reduce_fov_both_0.5"

CPC6_CORRUPTIONS="lidar_crosstalk_noise_severity1 lidar_crosstalk_noise_severity2 lidar_crosstalk_noise_severity3 lidar_crosstalk_noise_severity4 lidar_crosstalk_noise_severity5 impulse_noise_severity1 impulse_noise_severity2 impulse_noise_severity3 impulse_noise_severity4 impulse_noise_severity5 add_gaussian_noise_std0.01_outRate0.03_outStd0.2 add_gaussian_noise_std0.01_outRate0.05_outStd0.2 add_gaussian_noise_std0.02_outRate0.03_outStd0.2 add_gaussian_noise_std0.02_outRate0.05_outStd0.2"

run_one () {
  local run_name="$1"
  local frac="$2"
  local corruptions="$3"

  mkdir -p logs

  local tag="${run_name}_plus_${frac}"

  echo "Running ${tag} at $(date)"

  PYTHONPATH=src python3 run_full_pipeline.py \
    --data-root "$DATA_ROOT" \
    --corruptions $corruptions \
    --corruption-sample-frac "$frac" \
    --feature-set corruption_plus \
    --run-name "${tag}" \
    2>&1 | tee "logs/${tag}.log"

  echo "Finished ${tag} at $(date)"
  echo
}

for frac in 0.1; do
  run_one "Final_f1_CPC3" "$frac" "$CPC3_CORRUPTIONS"
  run_one "Final_f1_CPC4" "$frac" "$CPC4_CORRUPTIONS"
  run_one "Final_f1_CPC6" "$frac" "$CPC6_CORRUPTIONS"
  done

echo "ALL RUNS FINISHED"