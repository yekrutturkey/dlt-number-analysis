# v0.5.3 分阶段性能基准

运行范围：仅目标期 08009、seed=20260000、fast profile；这不是正式回测。
未运行100期、完整 development、360组消融、calibration、final holdout或多种子实验。

## 阶段状态

- prepare: completed
- vectorized_full: completed
- object B2 bank=50: completed
- object B2 bank=100: completed
- object B2 bank=250: completed

## 实测耗时与内存

- preparation（实测）：3.771757s
- 完整 B2-B6 向量路径（实测）：0.114040s
- 候选生成（实测）：1.723522s
- 银行生成（实测）：0.674384s
- CandidateArrayBundle 转换（实测）：0.097134s
- FeasiblePortfolioIndexBank 构建（实测）：0.032714s
- 完整银行：size=1280；acceptance_rate=0.256。
- B2 向量路径（同口径实测）：0.015129s
- 向量路径平均每策略：0.022808s
- 向量评分吞吐：56120.90547383707 Portfolio/s。
- object B2 bank=50（实测）：1.357268s；独立进程峰值内存=589.625 MB；一致=True。
- object B2 bank=100（实测）：4.021178s；独立进程峰值内存=589.6484375 MB；一致=True。
- object B2 bank=250（实测）：7.416927s；独立进程峰值内存=589.7265625 MB；一致=True。
- vectorized_full 独立进程峰值内存：589.46484375 MB。
- 内存口径：memory_scope=isolated_subprocess_peak；各路径独立子进程。

## 对象完整银行外推

以下均为估算，不是真实完成时间：
- 拟合规模：[50, 100, 250]
- slope：0.028530609076954615 秒/Portfolio
- intercept：0.4610430230722377 秒
- R²：0.9559637051968202
- optimistic：35.960253s
- central：36.980223s
- conservative：38.000192s

## 评估重采样开销

- raw: status=completed；0.000104s；相对 raw=1.000x
- full: status=completed；0.111250s；相对 raw=1073.838x
- reduced: status=completed；0.043711s；相对 raw=421.923x

## 结论边界

- 性能命令累计：43.453s / 480s。
- 8分钟停止条件触发：False。
- 实测/估算加速比：2444.2946513521492（estimated）。
- 加速比口径：B2完整对象银行线性外推 / B2完整银行向量实测；不是对象实测全量时间。
- 结果一致性：50/100/250 子银行的 entry、5注、core/support 与六项评分均一致。
- 已确认瓶颈：对象路径会重新物化完整候选评分对象，并逐Portfolio构造Pydantic结果。
- 潜在瓶颈：准备载荷反序列化、对象候选重评分及单进程Pydantic构造。
- 下一步是否适合20期基准：当前单期命令约49秒，20期在现有8分钟预算下不适合；需先减少每阶段子进程载荷重建开销。本轮不会运行20期。

> 评分不等于真实中奖概率，彩票开奖结果是随机事件。
