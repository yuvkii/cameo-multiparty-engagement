# Shrinkage and calibration (auto-generated)

## 1. Variance compression

within-session sd: true 39.4, predicted 23.8, mean ratio 0.606 (range 0.47-0.73)
between-session sd of session means: true 13.9, predicted 8.3, ratio 0.600
per-fold bias vs that fold's true mean: r = -0.800 (p = 0.0006), slope -0.650, R^2 0.640
predicted mean = 0.350 x true mean + 32.6; fixed point 50.1 (corpus mean 58.3)

## 2. Post-hoc recalibration, leave-one-fold-out estimated

| fold | CCC before | CCC after | MAE before | MAE after | bias before | bias after |
|---|---|---|---|---|---|---|
| 03_20 s1 | 0.510 | 0.590 | 25.08 | 24.10 | +15.8 | +17.5 |
| 03_20 s3 | 0.489 | 0.589 | 29.92 | 26.28 | -19.1 | -19.8 |
| 03_20 s4 | 0.525 | 0.600 | 29.80 | 27.47 | -16.6 | -20.2 |
| 03_26 s1 | 0.743 | 0.826 | 21.16 | 15.31 | +1.8 | +2.5 |
| 03_26 s2 | 0.537 | 0.606 | 28.31 | 25.07 | +22.2 | +22.6 |
| 03_26 s3 | 0.647 | 0.729 | 25.65 | 18.84 | -9.5 | -13.2 |
| 03_26 s4 | 0.649 | 0.738 | 21.04 | 14.93 | -1.6 | -13.3 |
| 03_26 s5 | 0.683 | 0.799 | 23.23 | 16.79 | -1.4 | +1.6 |
| 03_26 s6 | 0.745 | 0.820 | 21.88 | 14.06 | -1.5 | +0.3 |
| 05_15 s1 | 0.504 | 0.552 | 27.44 | 26.80 | +9.1 | +14.1 |
| 05_15 s2 | 0.804 | 0.898 | 19.02 | 10.39 | -6.2 | -0.6 |
| 05_15 s4 | 0.737 | 0.818 | 19.73 | 13.93 | -7.3 | -0.5 |
| 05_15 s5 | 0.782 | 0.803 | 16.87 | 16.97 | +5.3 | +7.6 |
| 05_15 s6 | 0.724 | 0.829 | 20.76 | 14.13 | -11.4 | -6.4 |

| metric | argmax class (as trained) | expected-value score | recalibrated score |
|---|---|---|---|
| 4-level accuracy | 0.556 | 0.328 | 0.571 |
| 4-level macro-F1 | 0.468 | 0.311 | 0.473 |
| binary accuracy | 0.811 | 0.815 | 0.813 |
| CCC | - | 0.649 | 0.728 |
| Spearman | - | 0.750 | 0.752 |
| MAE | - | 23.56 | 18.93 |
| mean abs. per-fold bias | - | 9.2 | 10.0 |

folds improved by recalibration: CCC 14/14, MAE 13/14
mean estimated scale factor 0.606
