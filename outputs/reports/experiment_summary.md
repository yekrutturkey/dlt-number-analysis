# 严格策略实验汇总

本报告将历史拟合、校准和最终留出结果分开；随机波动不会被解释为预测能力。

## 历史拟合（development）

| phase | experiment_id | observation_count | best_front_hits | best_total_hits | at_least_three_front_rate | at_least_2_plus_1_rate | unique_hit_concentration | any_prize_rate | average_prize | median_prize | roi | roi_period_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development | B0 | 1637 | 1.5693 | 2.0831 | 0.0574 | 0.1979 | 0.5629 |  |  |  |  | 0 |
| development | B1 | 101 | 1.6337 | 2.0495 | 0.0594 | 0.2079 | 0.5645 |  |  |  |  | 0 |
| development | B2 | 100 | 1.6800 | 2.3100 | 0.0800 | 0.3000 | 0.5521 |  |  |  |  | 0 |
| development | B3 | 100 | 1.6500 | 2.2100 | 0.1100 | 0.2300 | 0.5771 |  |  |  |  | 0 |
| development | B4 | 100 | 1.6500 | 2.2000 | 0.1000 | 0.2400 | 0.5383 |  |  |  |  | 0 |
| development | B5 | 100 | 1.6100 | 2.2100 | 0.0500 | 0.2300 | 0.5523 |  |  |  |  | 0 |
| development | B6 | 100 | 1.6800 | 2.1900 | 0.0700 | 0.2000 | 0.5111 |  |  |  |  | 0 |
| development | B7 | 1637 | 1.6866 | 2.2382 | 0.0629 | 0.2217 | 0.4326 |  |  |  |  | 0 |
| development | B8 | 1637 | 1.6439 | 2.2169 | 0.0825 | 0.2401 | 0.4956 |  |  |  |  | 0 |

## 校准（calibration）

尚未执行或没有可用观测。

## 最终留出（final holdout）

尚未执行或没有可用观测。

## 与 B1 约束匹配随机基线的配对比较

| phase | experiment_id | baseline_id | metric | paired_observation_count | mean_paired_difference | bootstrap_95_lower | bootstrap_95_upper | permutation_p_value | holm_adjusted_p_value | benjamini_hochberg_adjusted_p_value | holm_significant_advantage | benjamini_hochberg_significant_advantage | statistically_significant_advantage | inference_method |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development | B0 | B1 | best_front_hits | 101 | -0.0990 | -0.2673 | 0.0693 | 0.2831 | 1.0000 | 0.9223 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | B0 | B1 | best_total_hits | 101 | 0.0198 | -0.1584 | 0.2079 | 0.9184 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | B0 | B1 | at_least_three_front_rate | 101 | -0.0297 | -0.0891 | 0.0297 | 0.4983 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | B0 | B1 | at_least_2_plus_1_rate | 101 | -0.0099 | -0.1287 | 0.1089 | 1.0000 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | B0 | B1 | unique_hit_concentration | 101 | -0.0042 | -0.0560 | 0.0459 | 0.8732 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | B2 | B1 | best_front_hits | 100 | 0.0400 | -0.1300 | 0.2200 | 0.7491 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | B2 | B1 | best_total_hits | 100 | 0.2600 | 0.0600 | 0.4600 | 0.0204 | 0.7546 | 0.2040 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | B2 | B1 | at_least_three_front_rate | 100 | 0.0200 | -0.0500 | 0.0900 | 0.7946 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | B2 | B1 | at_least_2_plus_1_rate | 100 | 0.0900 | -0.0200 | 0.2000 | 0.1588 | 1.0000 | 0.8342 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | B2 | B1 | unique_hit_concentration | 100 | -0.0131 | -0.0598 | 0.0276 | 0.5625 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | B3 | B1 | best_front_hits | 100 | 0.0100 | -0.1600 | 0.1900 | 1.0000 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | B3 | B1 | best_total_hits | 100 | 0.1600 | -0.0302 | 0.3700 | 0.1474 | 1.0000 | 0.8342 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | B3 | B1 | at_least_three_front_rate | 100 | 0.0500 | -0.0300 | 0.1300 | 0.3347 | 1.0000 | 0.9564 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | B3 | B1 | at_least_2_plus_1_rate | 100 | 0.0200 | -0.1000 | 0.1400 | 0.8628 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | B3 | B1 | unique_hit_concentration | 100 | 0.0120 | -0.0393 | 0.0626 | 0.6449 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | B4 | B1 | best_front_hits | 100 | 0.0100 | -0.1800 | 0.2100 | 1.0000 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | B4 | B1 | best_total_hits | 100 | 0.1500 | -0.0600 | 0.3600 | 0.1968 | 1.0000 | 0.8342 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | B4 | B1 | at_least_three_front_rate | 100 | 0.0400 | -0.0400 | 0.1200 | 0.4577 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | B4 | B1 | at_least_2_plus_1_rate | 100 | 0.0300 | -0.0800 | 0.1500 | 0.7381 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | B4 | B1 | unique_hit_concentration | 100 | -0.0269 | -0.0753 | 0.0148 | 0.2296 | 1.0000 | 0.8347 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | B5 | B1 | best_front_hits | 100 | -0.0300 | -0.2000 | 0.1500 | 0.8196 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | B5 | B1 | best_total_hits | 100 | 0.1600 | -0.0200 | 0.3500 | 0.1148 | 1.0000 | 0.8342 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | B5 | B1 | at_least_three_front_rate | 100 | -0.0100 | -0.0600 | 0.0500 | 1.0000 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | B5 | B1 | at_least_2_plus_1_rate | 100 | 0.0200 | -0.0900 | 0.1302 | 0.8602 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | B5 | B1 | unique_hit_concentration | 100 | -0.0128 | -0.0572 | 0.0309 | 0.5751 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | B6 | B1 | best_front_hits | 100 | 0.0400 | -0.1400 | 0.2300 | 0.7319 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | B6 | B1 | best_total_hits | 100 | 0.1400 | -0.0500 | 0.3300 | 0.1722 | 1.0000 | 0.8342 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | B6 | B1 | at_least_three_front_rate | 100 | 0.0100 | -0.0600 | 0.0800 | 1.0000 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | B6 | B1 | at_least_2_plus_1_rate | 100 | -0.0100 | -0.1200 | 0.1000 | 1.0000 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | B6 | B1 | unique_hit_concentration | 100 | -0.0540 | -0.0943 | -0.0150 | 0.0114 | 0.4331 | 0.1520 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | B7 | B1 | best_front_hits | 101 | 0.0891 | -0.1089 | 0.2874 | 0.4435 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | B7 | B1 | best_total_hits | 101 | 0.1287 | -0.0594 | 0.3168 | 0.2086 | 1.0000 | 0.8342 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | B7 | B1 | at_least_three_front_rate | 101 | 0.0396 | -0.0396 | 0.1188 | 0.4579 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | B7 | B1 | at_least_2_plus_1_rate | 101 | -0.0693 | -0.1782 | 0.0396 | 0.2997 | 1.0000 | 0.9223 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | B7 | B1 | unique_hit_concentration | 101 | -0.1476 | -0.1918 | -0.1074 | 0.0002 | 0.0080 | 0.0080 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | B8 | B1 | best_front_hits | 101 | -0.0495 | -0.2376 | 0.1386 | 0.6713 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | B8 | B1 | best_total_hits | 101 | 0.0792 | -0.1089 | 0.2871 | 0.4779 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | B8 | B1 | at_least_three_front_rate | 101 | -0.0099 | -0.0693 | 0.0594 | 1.0000 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | B8 | B1 | at_least_2_plus_1_rate | 101 | -0.0198 | -0.1386 | 0.0990 | 0.8674 | 1.0000 | 1.0000 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |
| development | B8 | B1 | unique_hit_concentration | 101 | -0.0948 | -0.1396 | -0.0511 | 0.0004 | 0.0156 | 0.0080 | False | False | False | paired bootstrap plus paired sign-flip permutation test with Holm and Benjamini-Hochberg corrections |

本表是配对计算冒烟检查；不使用这 100 期选择参数，也不根据本表声明任何策略优势。

> 评分不等于真实中奖概率，彩票开奖结果是随机事件。
