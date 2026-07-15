# 严格策略实验汇总

本报告按共同目标期 cohort 分组；不同目标范围、种子或实验版本的平均值不会放入同一横向排名。随机波动不会被解释为预测能力。

## 完整 development 结果

| phase | cohort_id | experiment_id | experiment_version | observation_count | target_start_issue | target_end_issue | seed_count | best_front_hits | best_total_hits | at_least_three_front_rate | at_least_2_plus_1_rate | unique_hit_concentration | any_prize_rate | average_prize | median_prize | roi | roi_period_count | common_target_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development | development_full | B0 | v0.5-b0-v1 | 1637 | 08008 | 18109 | 1 | 1.5693 | 2.0831 | 0.0574 | 0.1979 | 0.5629 |  |  |  |  | 0 | 1637 |
| development | development_full | B7 | v0.5-b7-v1 | 1637 | 08008 | 18109 | 1 | 1.6866 | 2.2382 | 0.0629 | 0.2217 | 0.4326 |  |  |  |  | 0 | 1637 |
| development | development_full | B8 | v0.5-b8-v1 | 1637 | 08008 | 18109 | 1 | 1.6439 | 2.2169 | 0.0825 | 0.2401 | 0.4956 |  |  |  |  | 0 | 1637 |

## v0.5.1 100期 paired smoke 结果

| phase | cohort_id | experiment_id | experiment_version | observation_count | target_start_issue | target_end_issue | seed_count | best_front_hits | best_total_hits | at_least_three_front_rate | at_least_2_plus_1_rate | unique_hit_concentration | any_prize_rate | average_prize | median_prize | roi | roi_period_count | common_target_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development | v051_paired_smoke_100 | B1 | v0.5.1-b1-shared-bank-v1 | 100 | 08009 | 08108 | 1 | 1.6400 | 2.0500 | 0.0600 | 0.2100 | 0.5651 |  |  |  |  | 0 | 100 |
| development | v051_paired_smoke_100 | B2 | v0.5.1-b2-shared-bank-v1 | 100 | 08009 | 08108 | 1 | 1.6800 | 2.3100 | 0.0800 | 0.3000 | 0.5521 |  |  |  |  | 0 | 100 |
| development | v051_paired_smoke_100 | B3 | v0.5.1-b3-shared-bank-v1 | 100 | 08009 | 08108 | 1 | 1.6500 | 2.2100 | 0.1100 | 0.2300 | 0.5771 |  |  |  |  | 0 | 100 |
| development | v051_paired_smoke_100 | B4 | v0.5.1-b4-shared-bank-v1 | 100 | 08009 | 08108 | 1 | 1.6500 | 2.2000 | 0.1000 | 0.2400 | 0.5383 |  |  |  |  | 0 | 100 |
| development | v051_paired_smoke_100 | B5 | v0.5.1-b5-shared-bank-v1 | 100 | 08009 | 08108 | 1 | 1.6100 | 2.2100 | 0.0500 | 0.2300 | 0.5523 |  |  |  |  | 0 | 100 |
| development | v051_paired_smoke_100 | B6 | v0.5.1-b6-shared-bank-v1 | 100 | 08009 | 08108 | 1 | 1.6800 | 2.1900 | 0.0700 | 0.2000 | 0.5111 |  |  |  |  | 0 | 100 |

## 单期 correctness smoke 结果

| phase | cohort_id | experiment_id | experiment_version | observation_count | target_start_issue | target_end_issue | seed_count | best_front_hits | best_total_hits | at_least_three_front_rate | at_least_2_plus_1_rate | unique_hit_concentration | any_prize_rate | average_prize | median_prize | roi | roi_period_count | common_target_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development | single_correctness_smoke | B1 | v0.5.1-b1-shared-bank-v1 | 1 | 08008 | 08008 | 1 | 1.0000 | 2.0000 | 0.0000 | 0.0000 | 0.5000 |  |  |  |  | 0 | 1 |

## 校准（calibration）结果

尚未执行或没有可用观测。

## 最终留出（final holdout）结果

尚未执行或没有可用观测。

## 与 B1 约束匹配随机基线的配对比较

### v0.5.1 100期 paired smoke 结果

共同配对期数：100；配对键为 cohort_id、target_issue 和 seed。

| phase | cohort_id | target_start_issue | target_end_issue | common_target_count | experiment_id | baseline_id | metric | paired_observation_count | mean_paired_difference | bootstrap_95_lower | bootstrap_95_upper | permutation_p_value | holm_adjusted_p_value | benjamini_hochberg_adjusted_p_value | holm_significant_advantage | benjamini_hochberg_significant_advantage | statistically_significant_advantage | inference_method |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B2 | B1 | best_front_hits | 100 | 0.0400 | -0.1300 | 0.2200 | 0.7491 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B2 | B1 | best_total_hits | 100 | 0.2600 | 0.0600 | 0.4600 | 0.0204 | 0.4895 | 0.2549 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B2 | B1 | at_least_three_front_rate | 100 | 0.0200 | -0.0500 | 0.0900 | 0.7946 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B2 | B1 | at_least_2_plus_1_rate | 100 | 0.0900 | -0.0200 | 0.2000 | 0.1588 | 1.0000 | 0.7027 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B2 | B1 | unique_hit_concentration | 100 | -0.0131 | -0.0598 | 0.0276 | 0.5625 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B3 | B1 | best_front_hits | 100 | 0.0100 | -0.1600 | 0.1900 | 1.0000 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B3 | B1 | best_total_hits | 100 | 0.1600 | -0.0302 | 0.3700 | 0.1474 | 1.0000 | 0.7027 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B3 | B1 | at_least_three_front_rate | 100 | 0.0500 | -0.0300 | 0.1300 | 0.3347 | 1.0000 | 0.9298 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B3 | B1 | at_least_2_plus_1_rate | 100 | 0.0200 | -0.1000 | 0.1400 | 0.8628 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B3 | B1 | unique_hit_concentration | 100 | 0.0120 | -0.0393 | 0.0626 | 0.6449 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B4 | B1 | best_front_hits | 100 | 0.0100 | -0.1800 | 0.2100 | 1.0000 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B4 | B1 | best_total_hits | 100 | 0.1500 | -0.0600 | 0.3600 | 0.1968 | 1.0000 | 0.7027 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B4 | B1 | at_least_three_front_rate | 100 | 0.0400 | -0.0400 | 0.1200 | 0.4577 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B4 | B1 | at_least_2_plus_1_rate | 100 | 0.0300 | -0.0800 | 0.1500 | 0.7381 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B4 | B1 | unique_hit_concentration | 100 | -0.0269 | -0.0753 | 0.0148 | 0.2296 | 1.0000 | 0.7174 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B5 | B1 | best_front_hits | 100 | -0.0300 | -0.2000 | 0.1500 | 0.8196 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B5 | B1 | best_total_hits | 100 | 0.1600 | -0.0200 | 0.3500 | 0.1148 | 1.0000 | 0.7027 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B5 | B1 | at_least_three_front_rate | 100 | -0.0100 | -0.0600 | 0.0500 | 1.0000 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B5 | B1 | at_least_2_plus_1_rate | 100 | 0.0200 | -0.0900 | 0.1302 | 0.8602 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B5 | B1 | unique_hit_concentration | 100 | -0.0128 | -0.0572 | 0.0309 | 0.5751 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B6 | B1 | best_front_hits | 100 | 0.0400 | -0.1400 | 0.2300 | 0.7319 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B6 | B1 | best_total_hits | 100 | 0.1400 | -0.0500 | 0.3300 | 0.1722 | 1.0000 | 0.7027 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B6 | B1 | at_least_three_front_rate | 100 | 0.0100 | -0.0600 | 0.0800 | 1.0000 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B6 | B1 | at_least_2_plus_1_rate | 100 | -0.0100 | -0.1200 | 0.1000 | 1.0000 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B6 | B1 | unique_hit_concentration | 100 | -0.0540 | -0.0943 | -0.0150 | 0.0114 | 0.2849 | 0.2549 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |

该 cohort 仅用于正确性冒烟，不用于参数选择或策略优势声明。

> 评分不等于真实中奖概率，彩票开奖结果是随机事件。
