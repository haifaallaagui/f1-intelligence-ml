# F1 Intelligence ML - Evaluation Report

Generated: 2026-11-13T13:40:17+00:00

Test window: 2022-2026 (2278 driver-race rows).

All models are trained on 1995-2018, tuned on 2019-2021 and evaluated on 2022-2026.  The test window is never used for model selection.


## Winner prediction - before_qualifying

Selected model: `catboost`

| log_loss | brier | accuracy | top3_hit_rate | n_races |
| --- | --- | --- | --- | --- |
| 0.1181 | 0.0342 | 0.4821 | 0.7679 | 112 |

### Baselines

| baseline | log_loss | brier | accuracy | top3_hit_rate | n_races |
| --- | --- | --- | --- | --- | --- |
| championship_order | 0.1498 | 0.0410 | 0.4911 | 0.7857 | 112 |
| recent_form | 0.1429 | 0.0396 | 0.3571 | 0.6786 | 112 |
| career_form | 0.1669 | 0.0448 | 0.0268 | 0.5268 | 112 |

### Top features

| feature | importance |
| --- | --- |
| constructor_last10_avg_finish | 1.9556 |
| driver_vs_teammate_quali_h2h_rate | 1.8174 |
| driver_season_avg_position_gain | 1.7682 |
| constructor_gap_to_leader | 1.7142 |
| driver_career_avg_classified_finish | 1.6199 |
| circuit_longitude | 1.5845 |
| driver_career_points | 1.4793 |
| driver_finish_trend_last5 | 1.4147 |
| driver_season_avg_quali | 1.3751 |
| driver_circuit_affinity | 1.3162 |


## Podium prediction - before_qualifying

Selected model: `catboost`

| log_loss | brier | precision | recall | f1 | exact_set_accuracy | n_races |
| --- | --- | --- | --- | --- | --- | --- |
| 0.2606 | 0.0817 | 0.5565 | 0.5565 | 0.5565 | 0.0536 | 112 |

### Baselines

| baseline | log_loss | brier | precision | recall | f1 | exact_set_accuracy | n_races |
| --- | --- | --- | --- | --- | --- | --- | --- |
| championship_order | 0.2961 | 0.0916 | 0.5863 | 0.5863 | 0.5863 | 0.1250 | 112 |
| recent_form | 0.3024 | 0.0940 | 0.5030 | 0.5030 | 0.5030 | 0.0625 | 112 |
| career_form | 0.4385 | 0.1177 | 0.3661 | 0.3661 | 0.3661 | 0.0000 | 112 |

### Top features

| feature | importance |
| --- | --- |
| driver_last10_avg_quali | 5.0922 |
| driver_last10_podium_rate | 3.9166 |
| driver_last5_podium_rate | 3.3938 |
| constructor_last3_avg_finish | 3.3656 |
| driver_season_avg_grid | 3.0233 |
| constructor_championship_position_norm | 2.3858 |
| constructor_season_avg_quali | 2.3609 |
| constructor_circuit_avg_quali | 2.2673 |
| constructor_last10_avg_finish | 2.0509 |
| constructor_last3_podium_rate | 2.0244 |


## Full classification (P1-P20) - before_qualifying

Selected model: `lightgbm_ranker`

| mae | mae_classified | rmse | r2 | spearman | ndcg_at_10 | ranking_accuracy | top1_accuracy | podium_precision_at_3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3.7761 | 3.7723 | 5.1835 | 0.2297 | 0.6084 | 0.7710 | 0.7365 | 0.3571 | 0.5149 |

### Baselines

| baseline | mae | mae_classified | rmse | r2 | spearman | ndcg_at_10 | ranking_accuracy | top1_accuracy | podium_precision_at_3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| championship_order | 3.6497 | 3.6420 | 5.1309 | 0.2452 | 0.6167 | 0.8091 | 0.7416 | 0.4911 | 0.5863 |
| recent_form | 4.0184 | 4.0159 | 5.4571 | 0.1462 | 0.5652 | 0.7475 | 0.7170 | 0.3571 | 0.5060 |
| career_form | 4.9227 | 4.9115 | 6.2850 | -0.1325 | 0.4338 | 0.6057 | 0.6549 | 0.0268 | 0.3661 |

### Top features

| feature | importance |
| --- | --- |
| driver_last10_avg_gain | 79.0000 |
| driver_circuit_avg_position_gain | 76.0000 |
| driver_last10_avg_quali | 70.0000 |
| driver_season_avg_grid | 63.0000 |
| constructor_finish_trend_last5 | 60.0000 |
| driver_vs_teammate_avg_quali_delta | 59.0000 |
| driver_circuit_avg_laps | 55.0000 |
| driver_season_avg_laps | 52.0000 |
| constructor_circuit_avg_quali | 52.0000 |
| driver_finish_trend_last5 | 51.0000 |


## Driver performance - before_qualifying

Selected model: `lightgbm`

| finish_mae | finish_rmse | finish_r2 | points_mae | points_rmse | points_r2 |
| --- | --- | --- | --- | --- | --- |
| 3.8587 | 4.7731 | 0.3468 | 3.7022 | 5.7340 | 0.4494 |

### Top features

| feature | importance |
| --- | --- |
| driver_season_avg_laps | 53.0000 |
| driver_last3_avg_gain | 51.0000 |
| driver_circuit_affinity | 51.0000 |
| driver_points_trend_last5 | 49.0000 |
| constructor_season_avg_position_gain | 48.0000 |
| driver_last10_avg_gain | 48.0000 |
| driver_vs_teammate_avg_quali_delta | 47.0000 |
| driver_circuit_avg_position_gain | 46.0000 |
| driver_finish_trend_last5 | 46.0000 |
| driver_career_avg_laps | 45.0000 |


## Winner prediction - after_qualifying

Selected model: `xgboost`

| log_loss | brier | accuracy | top3_hit_rate | n_races |
| --- | --- | --- | --- | --- |
| 0.0871 | 0.0260 | 0.5982 | 0.9196 | 112 |

### Baselines

| baseline | log_loss | brier | accuracy | top3_hit_rate | n_races |
| --- | --- | --- | --- | --- | --- |
| championship_order | 0.1498 | 0.0410 | 0.4911 | 0.7857 | 112 |
| recent_form | 0.1429 | 0.0396 | 0.3571 | 0.6786 | 112 |
| career_form | 0.1669 | 0.0448 | 0.0268 | 0.5268 | 112 |
| grid_order | 0.1443 | 0.0399 | 0.5804 | 0.8750 | 112 |
| qualifying_order | 0.1418 | 0.0397 | 0.5804 | 0.8750 | 112 |

### Top features

| feature | importance |
| --- | --- |
| grid_position_norm | 0.0993 |
| grid_position | 0.0923 |
| constructor_last10_podium_rate | 0.0268 |
| driver_career_win_rate | 0.0208 |
| constructor_last5_podium_rate | 0.0170 |
| driver_championship_position | 0.0158 |
| driver_last5_podium_rate | 0.0141 |
| driver_season_avg_grid | 0.0132 |
| constructor_last3_podium_rate | 0.0126 |
| driver_last3_podium_rate | 0.0119 |


## Podium prediction - after_qualifying

Selected model: `catboost`

| log_loss | brier | precision | recall | f1 | exact_set_accuracy | n_races |
| --- | --- | --- | --- | --- | --- | --- |
| 0.2080 | 0.0656 | 0.6696 | 0.6696 | 0.6696 | 0.2321 | 112 |

### Baselines

| baseline | log_loss | brier | precision | recall | f1 | exact_set_accuracy | n_races |
| --- | --- | --- | --- | --- | --- | --- | --- |
| championship_order | 0.2961 | 0.0916 | 0.5863 | 0.5863 | 0.5863 | 0.1250 | 112 |
| recent_form | 0.3024 | 0.0940 | 0.5030 | 0.5030 | 0.5030 | 0.0625 | 112 |
| career_form | 0.4385 | 0.1177 | 0.3661 | 0.3661 | 0.3661 | 0.0000 | 112 |
| grid_order | 0.2743 | 0.0825 | 0.6607 | 0.6607 | 0.6607 | 0.2054 | 112 |
| qualifying_order | 0.2743 | 0.0834 | 0.6339 | 0.6339 | 0.6339 | 0.1786 | 112 |

### Top features

| feature | importance |
| --- | --- |
| grid_position_norm | 16.0304 |
| grid_position | 13.3257 |
| constructor_last3_avg_finish | 2.3985 |
| driver_career_avg_quali | 2.1496 |
| constructor_last5_avg_finish | 1.7179 |
| constructor_last10_podium_rate | 1.6688 |
| driver_last10_podium_rate | 1.6490 |
| driver_championship_position_norm | 1.6421 |
| constructor_season_avg_grid | 1.6016 |
| quali_position | 1.5963 |


## Full classification (P1-P20) - after_qualifying

Selected model: `xgboost_ranker`

| mae | mae_classified | rmse | r2 | spearman | ndcg_at_10 | ranking_accuracy | top1_accuracy | podium_precision_at_3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3.2678 | 3.2629 | 4.7100 | 0.3640 | 0.6776 | 0.8688 | 0.7756 | 0.5893 | 0.6756 |

### Baselines

| baseline | mae | mae_classified | rmse | r2 | spearman | ndcg_at_10 | ranking_accuracy | top1_accuracy | podium_precision_at_3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| championship_order | 3.6497 | 3.6420 | 5.1309 | 0.2452 | 0.6167 | 0.8091 | 0.7416 | 0.4911 | 0.5863 |
| recent_form | 4.0184 | 4.0159 | 5.4571 | 0.1462 | 0.5652 | 0.7475 | 0.7170 | 0.3571 | 0.5060 |
| career_form | 4.9227 | 4.9115 | 6.2850 | -0.1325 | 0.4338 | 0.6057 | 0.6549 | 0.0268 | 0.3661 |
| grid_order | 3.4943 | 3.4835 | 5.0613 | 0.2655 | 0.6291 | 0.8394 | 0.7518 | 0.5804 | 0.6607 |
| qualifying_order | 3.4328 | 3.4245 | 4.9237 | 0.3049 | 0.6483 | 0.8370 | 0.7593 | 0.5804 | 0.6339 |

### Top features

| feature | importance |
| --- | --- |
| constructor_championship_position | 0.0801 |
| grid_position_norm | 0.0252 |
| constructor_championship_position_norm | 0.0222 |
| grid_position | 0.0199 |
| driver_points_share_season | 0.0190 |
| driver_season_podium_rate | 0.0159 |
| constructor_last10_avg_finish | 0.0141 |
| driver_championship_position_norm | 0.0127 |
| constructor_points_share_season | 0.0124 |
| constructor_career_win_rate | 0.0119 |


## Driver performance - after_qualifying

Selected model: `lightgbm`

| finish_mae | finish_rmse | finish_r2 | points_mae | points_rmse | points_r2 |
| --- | --- | --- | --- | --- | --- |
| 3.3704 | 4.3439 | 0.4590 | 3.0934 | 4.7760 | 0.6180 |

### Top features

| feature | importance |
| --- | --- |
| grid_position_norm | 170.0000 |
| driver_season_avg_laps | 67.0000 |
| driver_last10_avg_gain | 66.0000 |
| quali_position_norm | 65.0000 |
| driver_circuit_avg_position_gain | 58.0000 |
| driver_last5_avg_gain | 58.0000 |
| quali_gap_to_pole_seconds | 56.0000 |
| constructor_circuit_avg_position_gain | 54.0000 |
| quali_gap_to_pole_pct | 52.0000 |
| driver_vs_teammate_avg_quali_delta | 51.0000 |


## Constructor performance - any

Selected model: `random_forest`

| points_mae | points_rmse | points_r2 | best_finish_mae | best_finish_rmse | best_finish_r2 |
| --- | --- | --- | --- | --- | --- |
| 5.3414 | 7.8797 | 0.6561 | 2.4938 | 3.2763 | 0.5895 |

### Baselines

| baseline | points_mae | points_rmse | points_r2 | best_finish_mae | best_finish_rmse | best_finish_r2 |
| --- | --- | --- | --- | --- | --- | --- |
| championship_order | 16.1728 | 19.9163 | -1.1968 | 3.5246 | 4.8266 | 0.1091 |
| recent_form | 21.6487 | 32.2838 | -4.7723 | 13.2539 | 16.0390 | -8.8375 |

### Top features

| feature | importance |
| --- | --- |
| constructor_last10_avg_points | 0.0723 |
| team_avg_driver_driver_last5_avg_points | 0.0693 |
| constructor_season_points_per_race | 0.0579 |
| team_max_driver_driver_form_points_weighted_last5 | 0.0578 |
| team_avg_driver_driver_season_points_per_race | 0.0557 |
| constructor_form_points_weighted_last5 | 0.0508 |
| team_max_driver_driver_last5_avg_points | 0.0432 |
| constructor_last5_avg_points | 0.0428 |
| team_avg_driver_driver_form_points_weighted_last5 | 0.0424 |
| team_min_driver_driver_last5_avg_points | 0.0376 |


## Notes and limitations

- Historical aggregates are updated only after each race (leakage-safe).
- Mode 1 (before qualifying) never uses qualifying, grid or sprint columns.
- Mode 2 (after qualifying) uses the actual grid when it is available, otherwise a flagged qualifying-order proxy.
- OpenF1 coverage begins in 2023 and FastF1 timing enrichment is optional; missing values remain missing.
- Championship projections use Monte-Carlo Plackett-Luce sampling and do not simulate the fastest-lap bonus point.
- Metrics are reported on the chronological test window; the baseline comparison is the honest reference point.