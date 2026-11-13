# F1 Intelligence ML - Evaluation Report

Generated: 2026-11-12T12:15:10+00:00

Test window: 2022-2026 (2010 driver-race rows).

All models are trained on 1995-2018, tuned on 2019-2021 and evaluated on 2022-2026.  The test window is never used for model selection.


## Winner prediction - before_qualifying

Selected model: `catboost`

| log_loss | brier | accuracy | top3_hit_rate | n_races |
| --- | --- | --- | --- | --- |
| 0.1445 | 0.0192 | 0.1339 | 0.1339 | 112 |

### Baselines

| baseline | log_loss | brier | accuracy | top3_hit_rate | n_races |
| --- | --- | --- | --- | --- | --- |
| championship_order | 0.1500 | 0.0169 | 0.1161 | 0.1607 | 112 |
| recent_form | 0.1498 | 0.0178 | 0.0893 | 0.1250 | 112 |
| career_form | 0.1565 | 0.0198 | 0.0000 | 0.1161 | 112 |


## Podium prediction - before_qualifying

Selected model: `catboost`

| log_loss | brier | precision | recall | f1 | exact_set_accuracy | n_races |
| --- | --- | --- | --- | --- | --- | --- |
| 0.3395 | 0.0846 | 0.3690 | 0.5074 | 0.4235 | 0.0089 | 112 |

### Baselines

| baseline | log_loss | brier | precision | recall | f1 | exact_set_accuracy | n_races |
| --- | --- | --- | --- | --- | --- | --- | --- |
| championship_order | 0.3769 | 0.0896 | 0.3780 | 0.5149 | 0.4318 | 0.0268 | 112 |
| recent_form | 0.4218 | 0.0938 | 0.3542 | 0.4821 | 0.4045 | 0.0357 | 112 |
| career_form | 0.5032 | 0.1139 | 0.2560 | 0.3527 | 0.2938 | 0.0000 | 112 |


## Full classification (P1-P20) - before_qualifying

Selected model: `xgboost_ranker`

| mae | mae_classified | rmse | r2 | spearman | ndcg_at_10 | ranking_accuracy | top1_accuracy | podium_precision_at_3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3.9751 | 3.3298 | 5.5711 | 0.1488 | 0.6515 | 0.7484 | 0.7531 | 0.1250 | 0.3714 |

### Baselines

| baseline | mae | mae_classified | rmse | r2 | spearman | ndcg_at_10 | ranking_accuracy | top1_accuracy | podium_precision_at_3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| championship_order | 4.0030 | 3.4026 | 5.6324 | 0.1300 | 0.6344 | 0.7296 | 0.7477 | 0.1161 | 0.4032 |
| recent_form | 4.3264 | 3.6724 | 5.9376 | 0.0331 | 0.5891 | 0.7138 | 0.7252 | 0.0893 | 0.3778 |
| career_form | 5.1423 | 4.5358 | 6.6674 | -0.2192 | 0.4501 | 0.6060 | 0.6608 | 0.0000 | 0.2730 |


## Driver performance - before_qualifying

Selected model: `hist_gradient_boosting`

| finish_mae | finish_rmse | finish_r2 | points_mae | points_rmse | points_r2 |
| --- | --- | --- | --- | --- | --- |
| 3.8906 | 5.0685 | 0.2955 | 3.1157 | 4.5668 | 0.4967 |


## Winner prediction - after_qualifying

Selected model: `catboost`

| log_loss | brier | accuracy | top3_hit_rate | n_races |
| --- | --- | --- | --- | --- |
| 0.1431 | 0.0211 | 0.0893 | 0.1607 | 112 |

### Baselines

| baseline | log_loss | brier | accuracy | top3_hit_rate | n_races |
| --- | --- | --- | --- | --- | --- |
| championship_order | 0.1500 | 0.0169 | 0.1161 | 0.1607 | 112 |
| recent_form | 0.1498 | 0.0178 | 0.0893 | 0.1250 | 112 |
| career_form | 0.1565 | 0.0198 | 0.0000 | 0.1161 | 112 |
| grid_order | 0.1496 | 0.0171 | 0.0893 | 0.1607 | 112 |
| qualifying_order | 0.1484 | 0.0170 | 0.0893 | 0.1786 | 112 |


## Podium prediction - after_qualifying

Selected model: `catboost`

| log_loss | brier | precision | recall | f1 | exact_set_accuracy | n_races |
| --- | --- | --- | --- | --- | --- | --- |
| 0.3249 | 0.0722 | 0.4911 | 0.6741 | 0.5634 | 0.0446 | 112 |

### Baselines

| baseline | log_loss | brier | precision | recall | f1 | exact_set_accuracy | n_races |
| --- | --- | --- | --- | --- | --- | --- | --- |
| championship_order | 0.3769 | 0.0896 | 0.3780 | 0.5149 | 0.4318 | 0.0268 | 112 |
| recent_form | 0.4218 | 0.0938 | 0.3542 | 0.4821 | 0.4045 | 0.0357 | 112 |
| career_form | 0.5032 | 0.1139 | 0.2560 | 0.3527 | 0.2938 | 0.0000 | 112 |
| grid_order | 0.3345 | 0.0765 | 0.4911 | 0.6786 | 0.5652 | 0.0179 | 112 |
| qualifying_order | 0.3555 | 0.0785 | 0.4732 | 0.6518 | 0.5438 | 0.0268 | 112 |


## Full classification (P1-P20) - after_qualifying

Selected model: `xgboost_ranker`

| mae | mae_classified | rmse | r2 | spearman | ndcg_at_10 | ranking_accuracy | top1_accuracy | podium_precision_at_3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3.5234 | 2.8834 | 5.1589 | 0.2701 | 0.7139 | 0.8145 | 0.7925 | 0.0714 | 0.5302 |

### Baselines

| baseline | mae | mae_classified | rmse | r2 | spearman | ndcg_at_10 | ranking_accuracy | top1_accuracy | podium_precision_at_3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| championship_order | 4.0030 | 3.4026 | 5.6324 | 0.1300 | 0.6344 | 0.7296 | 0.7477 | 0.1161 | 0.4032 |
| recent_form | 4.3264 | 3.6724 | 5.9376 | 0.0331 | 0.5891 | 0.7138 | 0.7252 | 0.0893 | 0.3778 |
| career_form | 5.1423 | 4.5358 | 6.6674 | -0.2192 | 0.4501 | 0.6060 | 0.6608 | 0.0000 | 0.2730 |
| grid_order | 3.8010 | 3.1988 | 5.5017 | 0.1699 | 0.6596 | 0.8032 | 0.7649 | 0.0893 | 0.5238 |
| qualifying_order | 3.7085 | 3.1099 | 5.3540 | 0.2138 | 0.6772 | 0.7923 | 0.7715 | 0.0893 | 0.5048 |


## Driver performance - after_qualifying

Selected model: `hist_gradient_boosting`

| finish_mae | finish_rmse | finish_r2 | points_mae | points_rmse | points_r2 |
| --- | --- | --- | --- | --- | --- |
| 3.5583 | 4.7983 | 0.3686 | 2.7187 | 4.1892 | 0.5765 |


## Constructor performance - any

Selected model: `lightgbm`

| points_mae | points_rmse | points_r2 | best_finish_mae | best_finish_rmse | best_finish_r2 |
| --- | --- | --- | --- | --- | --- |
| 4.7449 | 6.9227 | 0.5753 | 2.8328 | 3.8459 | 0.4332 |

### Baselines

| baseline | points_mae | points_rmse | points_r2 | best_finish_mae | best_finish_rmse | best_finish_r2 |
| --- | --- | --- | --- | --- | --- | --- |
| championship_order | 13.9279 | 16.6408 | -1.4539 | 3.9488 | 5.4339 | -0.1316 |
| recent_form | 17.3147 | 25.1434 | -4.6022 | 10.8010 | 12.6624 | -5.1445 |


## Notes and limitations

- Historical aggregates are updated only after each race (leakage-safe).
- Mode 1 (before qualifying) never uses qualifying, grid or sprint columns.
- Mode 2 (after qualifying) uses the actual grid when it is available, otherwise a flagged qualifying-order proxy.
- OpenF1 coverage begins in 2023 and FastF1 timing enrichment is optional; missing values remain missing.
- Championship projections use Monte-Carlo Plackett-Luce sampling and do not simulate the fastest-lap bonus point.
- Metrics are reported on the chronological test window; the baseline comparison is the honest reference point.