# v0.5.1 B1-B6 100期配对冒烟实验

- 阶段：development
- profile：fast
- seed：20260000
- 连续目标期：08009 至 08108
- 目标期数量：100
- 每期银行生成次数：1
- 每期银行复用次数：5
- 自动扩大搜索的目标期数：0
- 断点续跑状态验证：通过
- 重复主键拒绝验证：通过
- 已完成任务跳过验证：通过

## 可行银行分布

| 指标 | 最小值 | 中位数 | 均值 | 最大值 |
| --- | ---: | ---: | ---: | ---: |
| bank_size | 1280 | 1393.0 | 1392.67 | 1512 |
| acceptance_rate | 0.256000 | 0.278600 | 0.278534 | 0.302400 |

## B2-B6 相对 B1 的配对统计

| phase | cohort_id | target_start_issue | target_end_issue | common_target_count | experiment_id | baseline_id | metric | paired_observation_count | mean_paired_difference | bootstrap_95_lower | bootstrap_95_upper | permutation_p_value | holm_adjusted_p_value | benjamini_hochberg_adjusted_p_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B2 | B1 | best_front_hits | 100 | 0.04 | -0.13 | 0.22 | 0.7490501899620076 | 1.0 | 1.0 |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B2 | B1 | best_total_hits | 100 | 0.26 | 0.06 | 0.46 | 0.020395920815836834 | 0.489502099580084 | 0.2549490101979604 |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B2 | B1 | at_least_three_front_rate | 100 | 0.02 | -0.05 | 0.09 | 0.7946410717856429 | 1.0 | 1.0 |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B2 | B1 | at_least_2_plus_1_rate | 100 | 0.09 | -0.02 | 0.2 | 0.15876824635072986 | 1.0 | 0.7027165995372354 |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B2 | B1 | unique_hit_concentration | 100 | -0.013071428571999999 | -0.059836309524525004 | 0.027630357146649933 | 0.5624875024995001 | 1.0 | 1.0 |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B3 | B1 | best_front_hits | 100 | 0.01 | -0.16 | 0.19 | 1.0 | 1.0 | 1.0 |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B3 | B1 | best_total_hits | 100 | 0.16 | -0.030249999999999985 | 0.37 | 0.14737052589482103 | 1.0 | 0.7027165995372354 |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B3 | B1 | at_least_three_front_rate | 100 | 0.05 | -0.03 | 0.13 | 0.33473305338932213 | 1.0 | 0.9298140371925615 |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B3 | B1 | at_least_2_plus_1_rate | 100 | 0.02 | -0.1 | 0.14 | 0.8628274345130974 | 1.0 | 1.0 |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B3 | B1 | unique_hit_concentration | 100 | 0.011999999999999997 | -0.03931904761855001 | 0.06260178571794996 | 0.644871025794841 | 1.0 | 1.0 |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B4 | B1 | best_front_hits | 100 | 0.01 | -0.18 | 0.21 | 1.0 | 1.0 | 1.0 |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B4 | B1 | best_total_hits | 100 | 0.15 | -0.06 | 0.36 | 0.1967606478704259 | 1.0 | 0.7027165995372354 |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B4 | B1 | at_least_three_front_rate | 100 | 0.04 | -0.04 | 0.12 | 0.45770845830833834 | 1.0 | 1.0 |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B4 | B1 | at_least_2_plus_1_rate | 100 | 0.03 | -0.08 | 0.15 | 0.7380523895220956 | 1.0 | 1.0 |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B4 | B1 | unique_hit_concentration | 100 | -0.026857142857000006 | -0.075348809529225 | 0.014812500004474979 | 0.22955408918216358 | 1.0 | 0.7173565286942611 |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B5 | B1 | best_front_hits | 100 | -0.03 | -0.2 | 0.15 | 0.8196360727854429 | 1.0 | 1.0 |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B5 | B1 | best_total_hits | 100 | 0.16 | -0.02 | 0.35 | 0.11477704459108179 | 1.0 | 0.7027165995372354 |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B5 | B1 | at_least_three_front_rate | 100 | -0.01 | -0.06 | 0.05 | 1.0 | 1.0 | 1.0 |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B5 | B1 | at_least_2_plus_1_rate | 100 | 0.02 | -0.09 | 0.13024999999999864 | 0.8602279544091181 | 1.0 | 1.0 |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B5 | B1 | unique_hit_concentration | 100 | -0.012809523807999997 | -0.057191071429149996 | 0.03090773809939998 | 0.5750849830033993 | 1.0 | 1.0 |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B6 | B1 | best_front_hits | 100 | 0.04 | -0.14 | 0.23 | 0.7318536292741452 | 1.0 | 1.0 |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B6 | B1 | best_total_hits | 100 | 0.14 | -0.05 | 0.33 | 0.17216556688662268 | 1.0 | 0.7027165995372354 |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B6 | B1 | at_least_three_front_rate | 100 | 0.01 | -0.06 | 0.08 | 1.0 | 1.0 | 1.0 |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B6 | B1 | at_least_2_plus_1_rate | 100 | -0.01 | -0.12 | 0.1 | 1.0 | 1.0 | 1.0 |
| development | v051_paired_smoke_100 | 08009 | 08108 | 100 | B6 | B1 | unique_hit_concentration | 100 | -0.05402380952400001 | -0.09431071428707502 | -0.015046428573750006 | 0.011397720455908818 | 0.28494301139772044 | 0.2549490101979604 |

本实验只验证配对计算和工程闭环，不使用这100期结果选择参数，也不声明策略优势。

> 评分不等于真实中奖概率，彩票开奖结果是随机事件。
