# BioTransit AI — CDR Final Teknik Özet

## 1) Holdout Metrikleri (rastgele %20 test)

| model                 |   accuracy |   balanced_accuracy |   macro_f1 |
|:----------------------|-----------:|--------------------:|-----------:|
| Random Forest         |      0.907 |               0.899 |      0.87  |
| Multivariate logistic |      0.887 |               0.889 |      0.845 |
| Simple threshold      |      0.407 |               0.557 |      0.377 |
| Dummy most-frequent   |      0.733 |               0.333 |      0.282 |

## 2) 5-Fold Cross-Validation

| model                   |   accuracy_mean |   accuracy_sd |   balanced_accuracy_mean |   balanced_accuracy_sd |   macro_f1_mean |   macro_f1_sd |
|:------------------------|----------------:|--------------:|-------------------------:|-----------------------:|----------------:|--------------:|
| Multivariate logistic   |           0.927 |         0.006 |                    0.91  |                  0.022 |           0.882 |         0.016 |
| Random Forest           |           0.915 |         0.025 |                    0.849 |                  0.074 |           0.845 |         0.06  |
| Decision tree (depth=3) |           0.813 |         0.033 |                    0.766 |                  0.035 |           0.73  |         0.038 |
| Dummy most-frequent     |           0.73  |         0.004 |                    0.333 |                  0     |           0.281 |         0.001 |

## 3) Leave-One-Scenario-Out (Genelleme Testi)

| held_out_scenario        |   n | classes_in_test   |   accuracy |   macro_f1_all_3_classes |
|:-------------------------|----:|:------------------|-----------:|-------------------------:|
| combined_stress          |  90 | HIGH, LOW, MEDIUM |      0.078 |                    0.068 |
| drone_like_vibration     |  60 | LOW               |      1     |                    0.333 |
| high_vibration           |  80 | LOW, MEDIUM       |      0.762 |                    0.288 |
| prolonged_temp_excursion |  80 | HIGH, LOW, MEDIUM |      0.312 |                    0.159 |
| shock_heavy              |  80 | LOW               |      1     |                    0.333 |
| short_temp_excursion     |  90 | LOW               |      0.911 |                    0.318 |
| stable_cold_chain        | 120 | LOW               |      1     |                    0.333 |

## 4) Permutation Feature Importance

| feature                    |   importance_mean |   importance_std |
|:---------------------------|------------------:|-----------------:|
| vibration_dose_g2_h        |            0.0531 |           0.0163 |
| vibration_minutes_above_2g |            0.0445 |           0.013  |
| max_accel_rms_g            |            0.0424 |           0.0112 |
| mean_accel_rms_g           |            0.0389 |           0.0138 |
| shock_count_ge_5g          |            0.0319 |           0.0101 |
| thermal_auc_above_8c_c_h   |            0.028  |           0.0188 |
| max_shock_g                |            0.0245 |           0.0127 |
| shock_count_ge_20g         |            0.0162 |           0.0126 |
| time_above_8c_min          |            0.0115 |           0.0166 |
| temp_sd_c                  |            0.0111 |           0.021  |
| mean_temp_c                |            0.0075 |           0.0082 |
| dominant_vibration_freq_hz |            0.0051 |           0.0078 |
| mean_shock_g               |            0.0041 |           0.0051 |
| max_temp_c                 |            0.0019 |           0.0174 |
| duration_h                 |            0.0012 |           0.0086 |

## Görseller

- `FIG1_scenario_holdout_collapse.png` — combined_stress çöküşü
- `FIG2_permutation_importance.png` — termal/mekanik özellik önemi
