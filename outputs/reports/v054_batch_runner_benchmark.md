# v0.5.4 正式批量 Runner 10 期性能验证

## 运行范围

仅运行 08009–08018、B1–B6、seed=20260000、fast、numpy_vectorized、raw_observation、单进程。未运行正式历史实验。

完成目标期：10/10。
5 分钟主动停止条件是否触发：False。

raw_observation 仅计算已生成 PredictionRecord 的直接命中和可用历史奖金；它完全跳过随机 Monte Carlo、百分位、Bootstrap 和策略汇总。full_resampling 才适用于最终正式统计报告。

## 每期完整耗时

| target_issue | seconds |
|---|---:|
| 08009 | 3.274152 |
| 08010 | 3.288471 |
| 08011 | 3.112352 |
| 08012 | 3.130496 |
| 08013 | 3.182391 |
| 08014 | 2.976745 |
| 08015 | 2.867407 |
| 08016 | 3.111593 |
| 08017 | 3.096453 |
| 08018 | 3.048828 |

## 汇总与外推

实测 target_total_seconds：最小 2.867407s；中位 3.111973s；平均 3.108889s；P90 3.275584s；最大 3.288471s；标准差 0.120411s。
完整 development 串行中心外推（估算，不是实测）：5089.25s；保守外推（估算）：6702.66s。
包含子进程启动与汇总控制的实测墙钟时间：37.386s。

## 各阶段耗时汇总

| stage | mean_seconds | p90_seconds | maximum_seconds |
|---|---:|---:|---:|
| b1_sampling_seconds | 0.060163 | 0.067122 | 0.267479 |
| bank_generation_seconds | 0.669231 | 0.827899 | 0.833920 |
| candidate_array_conversion_seconds | 0.103700 | 0.114771 | 0.123171 |
| candidate_generation_seconds | 1.768971 | 1.866951 | 1.933567 |
| checkpoint_write_seconds | 0.040706 | 0.044958 | 0.060737 |
| final_object_construction_seconds | 0.001287 | 0.001500 | 0.001589 |
| history_slice_seconds | 0.000352 | 0.000424 | 0.000462 |
| observation_serialization_seconds | 0.000542 | 0.000625 | 0.000854 |
| portfolio_index_bank_seconds | 0.036521 | 0.037891 | 0.038055 |
| raw_evaluation_seconds | 0.000286 | 0.000338 | 0.000411 |
| score_view_seconds | 0.105878 | 0.112799 | 0.119664 |
| target_total_seconds | 3.108889 | 3.275584 | 3.288471 |
| vectorized_scoring_seconds | 0.000554 | 0.000696 | 0.000715 |
当前主要瓶颈（按每期平均实测耗时）：candidate_generation_seconds (1.769s)、bank_generation_seconds (0.669s)、score_view_seconds (0.106s)。

## 工程判断

本地运行：本地串行可行，预计约数小时。
AutoDL：暂不需要AutoDL。
是否进入完整 development：True。

潜在瓶颈：完整历史窗口增长可能增加候选结构评分成本；银行接受率变化可能触发search_trials 扩展。本轮未据 10 期结果修改策略参数。

未运行：object_reference、20/100 期、完整 development、360 组消融、calibration、final holdout、多 seed、多进程基准。

> 风险声明：评分不等于真实中奖概率，彩票开奖结果是随机事件。
