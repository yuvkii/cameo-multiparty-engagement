# Corpus numbers (auto-generated)

## 1. Session inventory

| date | session | fps | duration (s) | annotated frames | used from (s) | reconciliation | graphs | person-frames | mean label |
|---|---|---|---|---|---|---|---|---|---|
| 03_20 | 1 | 30 | 141 | 4226 | 0 | rank | 671 | 2013 | 33.1 |
| 03_20 | 3 | 30 | 175 | 5242 | 0 | rank | 849 | 2547 | 68.7 |
| 03_20 | 4 | 30 | 232 | 6964 | 18 | tracker | 1056 | 3079 | 59.2 |
| 03_26 | 1 | 30 | 175 | 5260 | 48 | rank | 595 | 1760 | 49.4 |
| 03_26 | 2 | 30 | 101 | 3022 | 0 | rank | 452 | 1356 | 25.7 |
| 03_26 | 3 | 30 | 201 | 6043 | 0 | rank | 958 | 2874 | 53.2 |
| 03_26 | 4 | 30 | 36 | 1065 | 0 | rank | 151 | 450 | 29.7 |
| 03_26 | 5 | 30 | 264 | 7930 | 0 | tracker | 1287 | 3830 | 54.9 |
| 03_26 | 6 | 30 | 402 | 12070 | 0 | rank | 1968 | 5904 | 59.0 |
| 05_15 | 1 | 30 | 196 | 5885 | 0 | rank | 712 | 2136 | 47.4 |
| 05_15 | 2 | 30 | 311 | 9317 | 0 | rank | 1467 | 4401 | 65.1 |
| 05_15 | 4 | 30 | 330 | 9912 | 0 | rank | 1484 | 4364 | 68.7 |
| 05_15 | 5 | 30 | 337 | 10107 | 180 | rank | 739 | 2203 | 50.2 |
| 05_15 | 6 | 30 | 546 | 16385 | 0 | rank | 2661 | 7981 | 68.9 |
| 05_14 | 1 | 60 | 119 | 7166 | 0 | rank | 503 | 1509 | 28.0 |
| 05_14 | 2 | 60 | 398 | 23893 | 15 | rank | 1831 | 5491 | 56.4 |
| 05_14 | 3 | 60 | 621 | 37277 | 20 | tracker | 2950 | 8427 | 41.0 |
| 05_14 | 4 | 60 | 684 | 41038 | 27 | rank | 3212 | 9636 | 65.1 |
| 05_14 | 5 | 60 | 607 | 36439 | 17 | rank | 2900 | 8700 | 65.0 |
| 05_14 | 6 | 60 | 690 | 41424 | 4-686 | rank | 3301 | 9901 | 73.7 |

**Totals**: 29747 graphs, 88562 person-frames, 871,995 raw frame-level ratings before subsampling, 109 min of annotated footage, 20 sessions used of 20 annotated.

## 2. Label distribution

| date | n person-frames | mean | sd | %disengaged | %low | %medium | %high |
|---|---|---|---|---|---|---|---|
| 03_20 | 7639 | 55.5 | 40.8 | 29.6 | 15.3 | 14.5 | 40.6 |
| 03_26 | 16174 | 52.4 | 43.9 | 37.1 | 11.6 | 8.7 | 42.6 |
| 05_15 | 21085 | 63.9 | 39.1 | 22.2 | 12.6 | 15.9 | 49.3 |
| 05_14 | 43664 | 60.0 | 45.0 | 33.3 | 6.6 | 6.7 | 53.4 |
| **all** | 88562 | 59.1 | 43.3 | 31.1 | 9.7 | 9.9 | 49.3 |

## 3. Cross-participant engagement correlation versus lag

| lag (s) | mean r | sd across sessions | min | max |
|---|---|---|---|---|
| 0 | 0.450 | 0.319 | -0.286 | 0.793 |
| 0.5 | 0.446 | 0.316 | -0.287 | 0.785 |
| 1 | 0.436 | 0.309 | -0.289 | 0.777 |
| 2 | 0.400 | 0.299 | -0.292 | 0.759 |
| 3 | 0.362 | 0.289 | -0.297 | 0.734 |
| 5 | 0.292 | 0.260 | -0.303 | 0.676 |
| 10 | 0.179 | 0.218 | -0.315 | 0.493 |

## 4. Anchoring-bias check (max |r| between participants within a session, lag 0)

mean 0.450, max 0.793 (03_20 s3), threshold for re-annotation 0.90, sessions above threshold: 0

## 5. Feature coverage and cross-date feature scale

(node feature layout: 0-3 gaze one-hot, 4-5 centre, 6-7 size, 8 of_availability, 9 face_conf, 10-11 gaze angles, 12-19 AU, 20-27 emotion one-hot, 28 mouth motion)

| date | facial coverage | mean AU activation | mean mouth motion | gaze at robot | gaze at peer | gaze elsewhere |
|---|---|---|---|---|---|---|
| 03_20 | 91.6% | 0.069 | 9.66 | 78.6% | 11.1% | 10.3% |
| 03_26 | 91.9% | 0.075 | 7.18 | 68.7% | 18.3% | 12.9% |
| 05_15 | 92.4% | 0.165 | 7.21 | 64.0% | 24.7% | 11.3% |
| 05_14 | 95.4% | 0.132 | 3.28 | 68.7% | 17.6% | 13.7% |

## 6. Graph sizes

| nodes per graph | count | share |
|---|---|---|
| 1 | 9 | 0.0% |
| 2 | 661 | 2.2% |
| 3 | 29077 | 97.7% |
