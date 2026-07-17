# dlt-number-analysis

## v0.5.7.1：rename/copy 双端点审计与人工恢复事务

Git `status --porcelain=v1 -z` 的 rename/copy 记录按 Git 实际顺序解析：第一条路径是
目标路径，紧随其后的第二条路径是原始路径。审计同时检查 source 和 destination；任一
端点位于受保护的源代码、配置或冻结数据范围，或者任一生成文件端点未被本次 CLI
明确授权，正式执行都会被阻断。审计报告以 `source -> destination` 保留两个端点，
不会把移动到授权输出目录当作来源路径已经安全。

stale `RUNNING.lock` 清理和 schema_v4 `CURRENT` repair 现在都先持久化
`status=prepared` 的独立审计，再改变锁或指针，成功后原子更新为
`status=completed`；操作失败尽量更新为 `status=failed`。如果状态已经改变而最终审计
更新中断，prepared 文件仍保留。可使用以下只读模式检查 prepared、completed、failed
和不一致操作；它不会获取运行锁、修改 CURRENT、删除锁或生成候选：

```powershell
uv run python scripts/run_v05_experiments.py --audit-manual-operations `
  --target-issues <issue> --experiment-ids B1
```

清理 stale lock 前必须先确认原进程已经退出，再显式提供原因。首次 append 如果在完整
generation rename 后、CURRENT 创建前中断，会留下 valid orphan 且没有 CURRENT：正式
读取不会自动选择它，普通 resume 的 preflight 会因无效/缺失 CURRENT 阻断。恢复顺序为：

```powershell
uv run python scripts/run_v05_experiments.py --audit-generations `
  --target-issues <issue> --experiment-ids B1

uv run python scripts/run_v05_experiments.py --repair-current `
  --repair-generation-id <validated-generation-id> `
  --repair-reason "人工核验后的恢复原因" `
  --target-issues <issue> --experiment-ids B1
```

不得手工创建或编辑 `CURRENT`，不得因为存在 orphan 就按 mtime 或名称选择“最新”
generation。若 `--audit-manual-operations` 报告 prepared 但未 completed，应人工核对锁是否
仍存在、CURRENT 是否已经指向 proposed generation，并保留审计后再决定下一步；工具不会
自动修复这些状态。

## v0.5.7：generation 事务、强制 preflight 与中断恢复

正式结果默认写入 `outputs/experiments/schema_v4/`。逻辑分区仍包含 phase、实验版本、
run context、execution config、cohort 和 seed；分区内部改为不可变 generation：

```text
seed_<seed>/
  CURRENT
  generations/<generation_id>/
    observations.parquet
    manifest.json
```

每次 append 都写一个完整的新 generation，校验 Parquet SHA、manifest SHA、身份、主键和
completed target 集后，才用 `os.replace` 原子切换 JSON `CURRENT`。`CURRENT` 保存
`storage_schema_version`、`generation_id`、`manifest_sha256`、`updated_at`；manifest 在
schema_v3 字段之外保存 generation/parent ID、generation 创建时间、观测数和
`committed=true`。旧 generation 不在原地修改，也不在 append 中删除。

未被 `CURRENT` 引用的 generation 是 orphan：正式读取不会选择它，也不会按 mtime
猜测版本。使用 `--audit-generations` 只读查看；需要恢复时，必须用
`--repair-current --repair-generation-id <id> --repair-reason <原因>` 显式选择一个已完整
验证的 generation。修复会留下独立审计记录。不得手工修改 Parquet、manifest 或
`CURRENT`。

普通执行与 smoke 都先写 preflight v2，并在 `ready=false` 时、调度任务构造之前停止。
Git 检查允许本次 CLI 精确声明的结果、preflight、scheduler、runtime、报告和日志路径
存在，以便中断后 resume；`src/`、`tests/`、`scripts/`、`config/`、`data/raw/`、
`.github/`、`.gitignore`、`pyproject.toml` 或 `uv.lock` 的变化始终阻断正式执行。
`--allow-dirty` 只放宽 smoke 的非源代码变化，不能绕过冻结数据和配置检查。

正式 development 的操作顺序是：先在干净提交上运行 `--preflight-only`，审阅精确的
run/cohort 哈希与输出路径，再用完全相同参数普通执行。中断后重复同一命令；已由
`CURRENT` 提交的 target 会跳过，orphan 不会被自动采用。如存在 orphan 或损坏指针，
先执行只读 audit，再决定是否显式 repair。普通写入使用身份范围内的 `RUNNING.lock`
防止第二个写进程；异常留下的锁只能通过
`--clear-stale-run-lock --stale-lock-reason <原因>` 显式清理并生成审计。

报告、scheduler、runtime 和日志的默认位置在身份计算后解析为
`outputs/formal_runs/<run-hash>/<cohort-hash>/...`。V2、V3、legacy、v0.5.5 和 v0.5.6
审计产物保持只读兼容，不会被 V4 扫描、迁移或重写。

## v0.5.6：逻辑 cohort、schema_v3 与正式 preflight

v0.5.6 将完整科学目标集合与调度 task chunk 分离。控制器在分块之前创建
`CohortDefinitionIdentity`；`cohort_definition_sha256` 由身份版本、用途、阶段、cohort ID、
规范化完整期号列表及其计数和边界计算，不包含 chunk 大小、worker 数、任务顺序或续跑的
pending 集合。worker 只计算自己的 chunk，但所有输出行保留同一完整逻辑 cohort 身份。

新的正式结果默认写入：

`outputs/experiments/schema_v3/{phase}/{experiment_id}/{experiment_version}/<run-hash>/<execution-hash>/<cohort-hash>/seed_<seed>/`

schema_v3 主键为
`experiment_id + experiment_version + execution_config_sha256 + cohort_definition_sha256 + phase + seed + target_issue`。
schema_v2 和 legacy 结果继续只读兼容，不自动迁移、重写或参与 schema_v3 正式报告。

正式长任务应先运行只读 preflight；正式模式拒绝脏工作区，只有 smoke 加
`--allow-dirty` 才能显式覆盖：

```powershell
uv run python scripts/run_v05_experiments.py --experiment-ids B1 B2 B3 B4 B5 B6 --preflight-only
```

report-only 必须精确指定一个 run context 和一个逻辑 cohort，不会扫描合并全部结果：

```powershell
uv run python scripts/run_v05_experiments.py --experiment-ids B1 B2 B3 B4 B5 B6 `
  --report-only `
  --run-context-sha256 <64位SHA256> `
  --cohort-definition-sha256 <64位SHA256>
```

v0.5.6 仅允许的验证入口分为一次实际 6 期双分块 smoke、一次只读续跑检查和一次精确
report-only；后两步不调用候选或银行生成：

```powershell
uv run python scripts/run_v056_cohort_smoke.py
uv run python scripts/run_v056_cohort_smoke.py --resume-check-only --chunk-size 2
uv run python scripts/run_v056_cohort_smoke.py --report-only `
  --run-context-sha256 <64位SHA256> `
  --cohort-definition-sha256 <64位SHA256>
```

## v0.5.5: formal experiment identity and schema_v2 isolation

Formal observations now use two audit identities. `run_context_sha256` binds the frozen
canonical history, concrete chronological split boundaries, phase, evaluation semantics,
expanded profile, requested Portfolio scoring path, bank limits, scoring implementation and
prize-data hashes. It deliberately excludes experiment specifications, target chunks, worker
count, host, wall-clock data and Git SHA. Each ExperimentSpec then receives
`execution_config_sha256 = sha256(run_context_sha256 + experiment_config_sha256)`.

New results are stored below
`outputs/experiments/schema_v2/{phase}/{experiment_id}/{experiment_version}/<run-hash>/<execution-hash>/seed_<seed>/`.
Every partition contains an atomically written Parquet file and a matching manifest. Resume
status requires the full ExperimentSpec, execution identity and run context; old v0.5.1 flat
partitions cannot complete a v0.5.5 task. The CLI never migrates legacy CSV implicitly.
`--migrate-legacy-observations` imports it only into `outputs/experiments/legacy_import` as
`legacy_unverified` and excludes it from formal statistics.

The bounded correctness entry point runs only the dynamically selected first eligible, median
and last development targets with B1-B6, seed 20260000, fast/raw/NumPy and one process:

```powershell
uv run python scripts/run_v055_identity_smoke.py
uv run python scripts/run_v055_identity_smoke.py --resume-check-only
```

This smoke validates identity, storage and read-only resume behavior only. It is not used for
parameter selection or a strategy-advantage claim. 评分不等于真实中奖概率，彩票开奖结果是随机事件。

## v0.5.4：正式批量 Runner 的 raw observation 模式

共享 B1–B6 Runner 现在显式支持 `raw_observation` 与 `full_resampling`。前者只对已经生成的
`PredictionRecord` 计算直接命中及当期可用奖金，完全跳过随机 Monte Carlo、百分位、Bootstrap、
策略汇总和单期 `run_rolling_backtest`；后者保持原有完整统计语义和 1000/1000 默认重采样。批量开发
实验推荐 raw 模式，final holdout 强制 full 模式。两种模式均写入 `evaluation_mode` 审计字段。

受限性能入口为：

```powershell
uv run python scripts/run_v054_batch_runner_benchmark.py
uv run python scripts/run_v054_batch_runner_benchmark.py --resume-check-only
```

该入口硬限制为 08009–08018、B1–B6、seed 20260000、fast、NumPy 向量化、raw、单进程，
并将结果与正式实验分区隔离在 `outputs/benchmarks/v054/`。第二条命令只校验并汇总 checkpoint，
不会重新生成候选号码或 Portfolio 银行。

26079 期的用户提供信息尚未经过两个独立官方/公开来源交叉核验，本版本不将其追加到冻结的 v0.5
历史快照；它只能在新的、完成来源核验的历史快照版本中加入。

中国体育彩票超级大乐透（以下简称“大乐透”）历史开奖数据校验、统计特征、策略比较与严格时间序列滚动回测项目。

> **重要声明：评分不等于真实中奖概率，彩票开奖结果是随机事件。**

本项目只用于数据工程、统计分析与回测研究，不承诺、暗示或声称能够预测中奖号码。

## v0.5.2：Portfolio 向量化评分与 cohort 报告

v0.5.2 保留 v0.5.1 的候选池、可行银行、随机种子和硬约束语义，只替换 B2–B6 和消融配置的
评分执行方式：

- `CandidateArrayBundle` 将一个 `CandidatePool` 按原顺序转换一次，使用只读整数/浮点 NumPy
  数组保存前后区号码、号码分、结构分、组合分及结构分类代码；票据号码身份映射不依赖
  `candidate_id`。
- `FeasiblePortfolioIndexBank` 为每个银行成员只保存 5 个候选索引，并预计算与评分配置无关的
  diversity、core concentration、structure coverage 和 repeat penalty。
- `CandidateScoreView` 只绑定号码分、结构分和组合分数组，不再为 B2–B6 或每个消融参数组合
  重新构造 10,000 个 `CandidateTicketScore`。
- `score_portfolio_bank_vectorized` 一次性计算全部银行成员，只为最终 entry 构造 5 张
  `PortfolioTicket`、一个 `PortfolioScoreBreakdown` 和一个 `PortfolioSelection`。
- 正式共享实验默认 `portfolio_scoring_method=numpy_vectorized`；
  `object_reference` 继续作为语义参考实现。B1 仍从银行按种子随机抽取，不受向量评分影响。
- 对象路径与数组路径的 B2–B6 测试在 entry hash、5注号码、core/support 和全部子评分上完全一致；
  candidate ID/顺序扰动及完全同分 tie-break 也保持确定性。

`experiment_summary.md` 现在按精确 target cohort 分组，分别展示完整 development、v0.5.1
100期 paired smoke、单期 correctness smoke、calibration 和 final holdout。配对检验只在相同
`cohort_id + target_issue + seed` 内进行，并明确保存起止期、共同期数、种子数和实验版本。

本轮严格停止了未能在单条 5 分钟内完成的三期对象/向量性能命令：workers=1 在 301 秒停止，
workers=2 在 281 秒停止，加上首次内存 API 失败约 10 秒，累计约 592 秒后不再运行实验。
因此 [v0.5.2 小型基准报告](outputs/reports/v052_vectorized_benchmark.md) 将完整加速比和峰值内存
标记为不可用，不从部分运行推断性能结论。后续基准已支持逐目标 checkpoint，但本轮不再重跑。

评分不等于真实中奖概率，彩票开奖结果是随机事件。

## v0.5.1：实验正确性与计算复用

v0.5.1 修正约束匹配随机基线并建立可中断恢复的共享计算路径：

- B1 不再调用 `optimize_portfolio`。`build_feasible_portfolio_bank` 只按号码和硬约束生成
  `FeasiblePortfolioBank`，B1 使用独立种子从银行均匀随机抽取；候选分数只在抽取后用于审计，
  不参与选择。银行身份不含 `candidate_id`，候选编号或顺序变化不会改变 B1 号码分布。
- 同一 `target_issue`、seed 和约束签名只生成一次银行，并保存 bank seed、bank size、
  acceptance rate、bank hash 和候选号集合哈希。B2–B6 与 B1 共享候选号、银行成员和约束，
  但 B2–B6 根据各自目标函数对同一个银行评分。
- 360 组消融采用目标期优先的共享路径：候选号与原始结构特征各计算一次，15 种
  window/decay 号码分通过 NumPy 批量计算，4 种结构权重组成评分矩阵，6 种
  pool-size/core-count 约束各生成一个银行。不会重复运行 360 次完整 Pipeline。
- 原始实验观测写入
  `outputs/experiments/{phase}/{experiment_id}/seed_{seed}.parquet`。五字段主键重复会拒绝，
  已完成目标期自动跳过，部分分区只追加缺失期次；三个时间阶段物理隔离。旧版 CSV 只作为
  一次性迁移来源，不再作为新结果写入目标。
- `ProcessPoolExecutor` 按目标期块或实验配置分片，默认进程数为
  `min(cpu_count - 1, 8)`；每个任务保存确定性子种子、worker 数量、主机、开始/结束时间、
  CPU/墙钟时间和失败任务。Python Portfolio 搜索不使用 ThreadPool。
- 历史审计新增年度期号缺口、开奖日期异常、官方分页重复、来源记录数和 verified prefix
  检查。真实缺失、分页重叠、来源不一致或前缀被修改均阻止正式实验；长假期等日期 warning
  只进入报告，不修改数据，也不阻断运行。
- 消融阶段固定为 screening（开发集每 5 期取 1 期、360 组、1 seed）、
  full development（前 30 组、3 seeds）、calibration（前 5–10 组、5 seeds）和一次性
  final holdout（冻结 1 组）。统计报告同时提供 Holm 与 Benjamini–Hochberg 修正。

本阶段只运行了 B1 单期正确性验证：目标期 `08008`、截止期 `08007`，银行含 1,191 个
唯一可行 Portfolio，5,000 次尝试的接受率为 23.82%。该单期结果仅验证执行链路和审计字段，
不能用于策略比较或显著性结论。完整历史范围的 B1–B6 和 360 组消融仍未运行。

### v0.5.1：B1–B6 100期配对冒烟

在不运行 360 组消融的前提下，已使用 `seed=20260000` 和 `fast` profile 对 development
中的连续目标期 `08009–08108` 运行 B1–B6，共保存 600 个严格向前观测：

- 每个目标期只生成 1 份候选号码和 1 个约束相同的可行 Portfolio 银行；B1–B6 的
  `candidate_numbers_hash` 一致，同约束的 `bank_hash` 一致，后续 5 个策略复用该银行。
- 银行最小容量门槛为 500；低于门槛时会用同一 bank seed 成倍扩大搜索预算，达到门槛后才继续。
  本次 100 期初始 5,000 次搜索均已达到门槛，没有实际触发扩容。
- `bank_size` 的最小值、中位数、均值、最大值分别为
  `1280 / 1393 / 1392.67 / 1512`；接受率对应为
  `0.256000 / 0.278600 / 0.278534 / 0.302400`。
- 第二次执行同一命令时 B1–B6 均自动跳过；真实重复主键试写被拒绝；六个分区对这 100 期的
  pending 集合均为空。
- 已生成 B2–B6 相对 B1 的 25 项配对 Bootstrap/置换检验结果，同时保存 Holm 与
  Benjamini–Hochberg 校正值。该冒烟结果不用于参数选择，也不用于声明任何策略优势。
- 本机 8 进程运行的正式 wall-clock 为 `6008.13` 秒，累计 CPU 时间为 `40413.48` 秒；
  多进程缩放明显非线性，后续应优先减少多评分 CandidatePool 的 Python 对象物化。

完整报告位于 `outputs/reports/v051_paired_smoke_100.md`，机器可读统计位于
`outputs/reports/v051_paired_smoke_comparisons.csv`。可重复入口：

```powershell
uv run python scripts/run_v051_paired_smoke.py --workers 8 --chunk-size 13
```

评分不等于真实中奖概率，彩票开奖结果是随机事件。

```powershell
# 单期或小批量运行；重复执行会自动跳过已完成主键
uv run python scripts/run_v05_experiments.py `
  --experiment-ids B1 --phase development --target-issues 08008 --workers 1

# 只从一个精确 schema_v3 run/cohort 分区汇总报告，不运行实验
uv run python scripts/run_v05_experiments.py --report-only `
  --run-context-sha256 <64位SHA256> `
  --cohort-definition-sha256 <64位SHA256>

# 非破坏性重建历史完整性报告
uv run python scripts/audit_verified_history.py
```

共享银行和 ProcessPool 指标汇总见 `outputs/reports/experiment_runtime.md`；每次任务的完整主机、
子种子、开始结束时间和失败列表保存在 `outputs/experiments/run_metadata/`。

## v0.5：完整历史与严格实验

v0.5 新增以下可审计边界：

- `data/raw/draws.csv` 已由两个独立公开来源逐期逐字段核对，覆盖 `07001–26078` 共
  2,896 期，当前冲突数为 0。
- 官方来源为[中国体育彩票历史接口](https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry)，
  独立来源为 [500.com 大乐透历史页](https://datachart.500.com/dlt/history/history.shtml)。
- `data/snapshots/` 保存不可覆盖的原始响应、抓取时间、URL、内容哈希和元数据；正式历史旁边的
  `draws.csv.verified.json` 绑定两份快照与规范历史哈希。
- `canonical_history_sha256` 使用固定列顺序、两位号码和 LF 换行，只依赖逻辑记录；
  `raw_file_sha256` 继续保留原文件身份。复盘会重新计算截止期前缀，历史被修改时拒绝执行。
- 正式 `generate-next` 和 `run-backtest` 默认只接受 verified history。研究用途必须显式传入
  `--allow-unverified-history`，生成 Artifact 会记录这一 override。
- `SourceFetcher`、`SourceSnapshot`、`normalize_source_draws`、`reconcile_draw_sources`、
  `resolve_conflict_record` 和 `write_verified_history` 构成与分析层解耦的双源导入流程；
  冲突永不自动裁决。
- `ExperimentSpec` 注册 B0–B8 基线，消融网格完整覆盖 3×5×4×3×2=360 组参数；
  development/calibration/final holdout 采用连续 60%/20%/20% 时间划分。
- 最终留出在运行前写入配置哈希，同一实验版本只允许一次正式结果；参数变化必须使用新版本。
- 统计比较使用配对指标差、配对 bootstrap 95% 区间和配对置换检验，并输出按年份、按种子和
  参数敏感性接口；不会用两个独立置信区间是否重叠来宣称优势。
- 性能基准记录总耗时、Python 分配峰值内存、候选生成/评分、Portfolio 搜索和可行组合数。
  当前完整历史实测中 NumPy 批量特征路径相对标量候选评分为 1.147x；4 线程没有稳定加速，
  因此 `final` 默认仍为 1 线程。

已完成的 development 历史结果包括 B0、B7、B8，各 1,637 个严格向前滚动观测；另有 B1
单期正确性验证。B1 完整基线、B2–B6 优化策略、360 组消融、calibration 和 final holdout 尚未完成，因此当前没有
统计依据声称任何策略显著优于约束匹配随机基线。详见 `outputs/reports/experiment_summary.md`。

可重复执行入口：

```powershell
# 联网获取两个来源、保存不可变快照并仅在完全一致时写 verified history
uv run python scripts/import_verified_history.py --end-issue 26078

# 运行小批量基线；新观测追加到分区 Parquet，已完成主键自动跳过
uv run python scripts/run_v05_experiments.py --experiment-ids B1 --target-issues 08008

# 比较 fast/standard/final、1/4 workers 和 NumPy/标量候选特征路径
uv run python scripts/run_v05_benchmark.py
```

## v0.4.1 基线能力

已实现：

- 历史开奖 CSV 的加载、结构校验、业务规则校验和校验后保存。
- 前后区静态数值特征、展示比例字符串及与上期重号数。
- 开奖来源、五注票据、预测日志、单注评估和复盘汇总 Pydantic 模型。
- 自第 26014 期生效的七级奖级 JSON 配置，以及完全配置驱动的评估函数。
- `max_coverage`、`core_rotation`、`hybrid_portfolio` 三种简单可替换策略。
- 每种策略固定五注、禁止完全重复，并保存模型版本、截止期号、生成时间、随机种子和全部参数。
- 严格扩展窗口滚动回测骨架及完全均匀随机五注 baseline。
- 第 26078 期用户提供的真实开奖、手工事前五注日志和复盘报告。
- `NumberScorer` 可替换接口，以及均匀、累计频率、近期加权频率和冷热混合评分。
- 只使用预测时点之前经验分布的动态票据结构评分。
- 带种子的至少 10000 注合法候选票池、逐注评分明细和 JSONL 持久化。
- 十元五注 Portfolio 约束优化、3 注核心/稳健票与 2 注探索票。
- 每个历史时点至少 1000 个随机种子的 baseline 分布、策略百分位和 Bootstrap 95% 置信区间。
- 按历史规则生效边界解析奖级；规则或实际奖金缺失时保留命中指标并禁用 ROI。
- `ScorerSpec` 绑定评分器名称、完整参数和实现版本，日志配置即实际调用配置。
- 不自动修正冲突的 `append_draws`、`reconcile_sources` 和数据质量阻断接口。
- 实时生成与滚动回测共用的 `PredictionPipeline`，以及四个 `dlt` CLI 子命令。
- Python 3.12/3.13 GitHub Actions 质量检查。
- `generate-next` 与 `evaluate-latest` 共用的 Artifact/PredictionRecord 解析闭环。
- 历史 CSV SHA-256、Git 工作区状态、差异哈希和 Pipeline 参数交叉校验。
- 默认至少 100 期历史，并为测试/研究短历史覆盖保存显式审计标记。
- `fast`、`standard`、`final` 三种可展开并允许显式覆盖的运行 profile。

v0.5.1 尚未完成：

- 旧规则时期的完整奖级配置和逐期实际奖金数据。
- 自动调度、开奖后触发和结果发布。
- B1–B6 完整历史回测、360 组消融、校准集选择和一次性最终留出结果。
- 复杂机器学习模型。

当前评分和优化器属于可替换的工程框架；部分轻量策略已有完整 development 观测，但尚未完成与
B1 约束匹配随机基线的配对比较，不代表真实中奖概率，也不声称任何策略优于随机。

## 环境与命令

要求 Python 3.12+，使用 [uv](https://docs.astral.sh/uv/) 管理依赖。

```powershell
uv sync --dev
uv run ruff format .
uv run ruff check .
uv run pytest
uv lock --check
git diff --check
```

运行测试时不需要网络访问。

## 项目结构

```text
dlt-number-analysis/
├── data/
│   ├── raw/                 # 原始历史开奖 CSV
│   └── processed/           # 校验或特征处理后的数据
├── config/                  # 奖级、奖金和票价配置
├── outputs/
│   ├── backtests/           # v0.5 以前的滚动回测结果与迁移来源
│   ├── experiments/         # 按 phase/experiment/seed 分区的追加式 Parquet 与运行元数据
│   ├── predictions/         # 预测日志
│   └── reports/             # 复盘报告
├── scripts/                 # 可重复执行的制品生成脚本
├── src/dlt_number_analysis/
│   ├── data/                # CSV 结构和数据校验
│   ├── analysis/            # 前后区统计特征
│   ├── scoring/             # 可替换号码评分和经验结构评分
│   ├── portfolio/           # 候选票池、评分明细和五注优化
│   ├── models/              # 来源、票据、预测和评估模型
│   ├── evaluation/          # 配置驱动的奖级评估与报告
│   ├── strategies/          # 五注组合策略及随机 baseline
│   ├── backtesting/         # 严格向前滚动回测
│   ├── pipeline/            # 实时与回测共用的端到端 Pipeline
│   ├── fetchers/            # 独立的数据抓取层（当前仅预留边界）
│   └── cli.py               # dlt 命令行入口
└── tests/
    ├── data/
    └── analysis/
```

抓取层只负责获取和转换外部数据，不得包含评分或回测逻辑；分析层只消费经过校验的规范数据，不直接访问外部数据源。

## 历史开奖 CSV 格式

文件编码使用 UTF-8，首行必须严格采用以下列名和顺序：

```csv
issue,draw_date,front_1,front_2,front_3,front_4,front_5,back_1,back_2
```

| 字段 | 类型/格式 | 规则 |
| --- | --- | --- |
| `issue` | 字符串 | 仅含数字；不得重复；保留前导零；按数值严格递增 |
| `draw_date` | `YYYY-MM-DD` | 合法日期；不得重复；按时间严格递增 |
| `front_1`…`front_5` | 整数 | 每个值为 1–35；同一期不得重复；必须严格升序 |
| `back_1`…`back_2` | 整数 | 每个值为 1–12；同一期不得重复；必须严格升序 |

不允许缺失值、额外列或不同的列顺序。加载时必须显式把 `issue` 作为字符串读取，避免丢失前导零。入口函数：

```python
from dlt_number_analysis.data import load_draws_csv, write_validated_draws_csv

draws = load_draws_csv("data/raw/history.csv")
write_validated_draws_csv(draws, "data/processed/history.validated.csv")
```

## 特征口径

`engineer_features` 会先校验输入，再按 CSV 中的时间顺序计算以下特征。原有比例字符串列继续保留用于展示，新增计数列供评分和统计直接使用。

| 输出列 | 定义 |
| --- | --- |
| `front_sum` | 5 个前区号码之和 |
| `front_span` | 最大前区号码减最小前区号码 |
| `front_odd_count` / `front_even_count` | 前区奇数/偶数个数 |
| `front_large_count` / `front_small_count` | 前区大号（18–35）/小号（1–17）个数 |
| `front_zone_1_count` | 前区一区（1–12）个数 |
| `front_zone_2_count` | 前区二区（13–24）个数 |
| `front_zone_3_count` | 前区三区（25–35）个数 |
| `odd_even_ratio` | 奇数个数与偶数个数，格式为 `奇:偶` |
| `large_small_ratio` | 大号（18–35）与小号（1–17）个数，格式为 `大:小` |
| `zone_ratio` | 一区（1–12）、二区（13–24）、三区（25–35）个数，格式为 `一区:二区:三区` |
| `consecutive_pair_count` | 排序后差为 1 的相邻号码对数；例如 `1,2,3` 计 2 对 |
| `same_tail_pair_count` | 个位数相同的号码对数；例如 `1,11,21` 计 3 对 |
| `repeat_from_previous_count` | 与上一期前区号码的交集大小；第一期为缺失值 |
| `back_sum` | 2 个后区号码之和 |
| `back_odd_count` / `back_even_count` | 后区奇数/偶数个数 |
| `back_large_count` / `back_small_count` | 后区大号（7–12）/小号（1–6）个数 |
| `back_consecutive_pair_count` | 两个后区号码是否构成连号，取值 0 或 1 |
| `back_repeat_from_previous_count` | 与上一期后区号码的交集大小；第一期为缺失值 |

```python
from dlt_number_analysis.analysis import engineer_features

featured = engineer_features(draws)
```

前后区第一期的重号数均使用 `pandas.NA`，因为“没有上期”与“有上期但重号数为 0”不是同一含义。只允许使用当前期及之前的数据计算某一期特征，不得使用未来开奖数据生成历史预测。

## 预测日志与评估

`PredictionRecord` 包含固定五注 `TicketRecord`，并强制记录：

- 模型/策略版本；
- 数据截止期号；
- 带时区的生成时间；
- 记录来源 `prediction_origin`；
- 随机种子；
- 完整生成参数；
- 固定声明：**评分不等于真实中奖概率，彩票开奖结果是随机事件。**

手工历史记录如果缺少原始生成时间，字段保留为 `null`，同时记录补录时间和缺失原因，不得猜测时间。代码生成的预测则必须提供带时区的 `generated_at` 和 `random_seed`。

奖级匹配与金额位于 [`config/prize_tiers.json`](config/prize_tiers.json)。当前配置适用于自第 26014 期开始的七级规则；一、二等奖为浮动奖，缺少具体期次奖金时保留为 `null`，回测不会用假金额计算 ROI。官方规则来源：[超级大乐透游戏规则](https://m.lottery.gov.cn/ksjz/m/yxgz_dlt/)。

## 第 26078 期手工预测复盘

- 开奖：前区 `02,13,20,25,32`，后区 `08,11`。
- 五注日志：[`outputs/predictions/26078_manual_chat.json`](outputs/predictions/26078_manual_chat.json)。
- 复盘报告：[`outputs/reports/26078_review.md`](outputs/reports/26078_review.md)。
- `prediction_origin=manual_chat`、`model_version=manual-v0`、`data_cutoff_issue=26077`。
- 这五注是用户提供的开奖前实际购买号码，不是当前代码生成结果。
- 第 2 注命中 2 个前区和 1 个后区；开奖前奖池为 8.18 亿元，按
  `pool_at_or_above_800m` 配置为七等奖 7 元。
- 总成本 10 元，总奖金 7 元，ROI 为 -30%。
- 五注号码池覆盖全部 5 个前区和 2 个后区，但号码池全覆盖不代表单注预测成功。

可重复生成复盘：

```powershell
uv run python scripts/generate_26078_review.py
```

## 可替换号码评分

`NumberScorer` 的统一输入是已经校验且截止于目标期之前的历史 `DataFrame`，分别对前区或后区完整号码空间返回工程评分：

- `uniform_score`：所有号码等分，用于无偏评分基准。
- `cumulative_frequency_score`：使用传入历史窗口的累计频率。
- `recency_weighted_frequency_score`：支持最近 10、30、100 期和可配置衰减系数。
- `hot_cold_blend_score`：混合近期加权热度与长窗口低频度。

评分器不会自行获取未来开奖。调用方必须只传入目标期之前的数据；滚动回测已在每个历史时点执行这个切片。

## 动态结构评分与候选票池

`fit_structure_profile` 从预测时点之前的历史特征拟合经验分布，不使用固定“合理区间”。连续概念 `front_sum`、`front_span`、`back_sum` 使用经验分位数中心性；离散概念使用带 Laplace 平滑的经验频率。奇偶只计算奇数个数，大小只计算大号个数，三区以完整 `zone_signature` 作为一个概念，避免把互补变量重复计权。13 个概念都有显式权重，权重和严格为 1；每项输出保存方法、参数、权重、原始分和加权分。

`generate_candidate_pool` 默认使用一个随机种子生成 10000 注互不相同的合法候选，为每注保存：

- `number_score`
- `structure_score`
- `combined_ticket_score`
- 全部静态结构特征和逐项结构分数
- 不可与实际调用分离的 `ScorerSpec`、原始号码分、历史截止期号、生成时间和随机种子

`write_candidate_score_details` / `load_candidate_score_details` 使用 JSONL 保存和读取完整评分明细。候选评分仅用于组合排序，评分不等于真实中奖概率。

## 五注 Portfolio 与原有组合策略

`optimize_portfolio` 在 10 元、5 注预算下执行带种子的可重复搜索。默认前区号码池范围为 16–21，目标池大小为 18；池大小得分按偏离目标的距离计算，不再简单奖励号码池越大越好。任意两注前区交集不超过 2，后区组合不重复，至少覆盖 3 个历史动态和值区间和 3 种三区结构，并输出 3 注核心/稳健票和 2 注探索票。

所有出现至少 2 次的前区号码都必须分类：2–3 个核心号码出现 2–3 次，2–4 个支撑号码最多出现 2 次，其余探索号码只出现 1 次，任意号码最多出现 3 次。`core_concentration` 同时评估核心覆盖、次数合规、三张稳健票覆盖和核心出现位置，不会仅因合法的核心轮转重复而扣分。目标函数保存以下明细：

- 单票分数
- 组合多样性
- 核心集中度
- 结构覆盖
- 过度重复惩罚

原有轻量策略继续保留：

- `max_coverage`：五注前区覆盖 25 个不同号码，后区覆盖 10 个不同号码，尽量减少跨票重复。
- `core_rotation`：五个核心前区号码各出现 2 或 3 次，其他位置优先低重复填充。
- `hybrid_portfolio`：3 注核心轮转与 2 注探索组合。
- `random_baseline`：完全均匀随机五注，不使用号码分数。

所有策略固定生成五注并由 `PredictionRecord` 禁止任意两注完全相同。

## 严格滚动回测

`run_rolling_backtest` 使用扩展窗口：预测索引为 `N` 的期开奖时，只把 `draws.iloc[:N]` 交给号码评分器和策略，第 N 期实际开奖只在预测生成后用于评估。禁止随机切分历史时间序列。

逐期输出：

- `best_front_hits`
- `best_back_hits`
- `best_total_hits`
- `ticket_hit_share`
- `unique_hit_concentration`
- `any_prize`
- `front_pool_coverage`
- `back_pool_coverage`
- `total_cost`
- `total_prize`
- `roi`，定义为 `(total_prize - total_cost) / total_cost`

跨期策略汇总新增：

- `at_least_three_front_rate`
- `at_least_2_plus_1_rate`
- `ticket_hit_share`
- `unique_hit_concentration`
- `average_prize`
- `median_prize`
- `longest_no_prize_streak`

完全随机五注对 8 项命中/覆盖指标生成至少 1000 个种子的分布；彩票号码标签对称，因此相同种子配置的分布在进程内缓存，不按期重复模拟。随机分布均值输出明确标为 Monte Carlo 误差的 95% 区间；策略跨历史期次的性能另用 Bootstrap 95% 区间，两者不可混用。百分位和区间只描述模拟或历史样本，不代表未来中奖概率。

### v0.3 指标迁移

`hit_concentration_ratio` 自 v0.4 更名为 `ticket_hit_share`，含义仍是“最佳单票命中数 / 所有票逐票命中数之和”。模型读取时兼容旧字段名，但新 JSON 只写出 `ticket_hit_share`。新增 `unique_hit_concentration = best_total_hits / (front_pool_coverage + back_pool_coverage)`；号码池无命中时为 0。后者按唯一命中号码计分，不会因为同一核心号码在多注中重复命中而人为降低集中度。

## 端到端 Pipeline 与 CLI

`PredictionPipeline` 的固定流程为：历史数据 → 号码评分 → 历史结构画像 → 至少 10000 注候选池 → 五注 Portfolio 优化 → `PredictionRecord`。`optimized_portfolio_strategy` 在滚动回测中接收目标期之前的完整扩展窗口；`generate-next` 使用同一个 Pipeline。

```powershell
uv run dlt validate-data
uv run dlt run-backtest
uv run dlt generate-next --target-issue 26079
uv run dlt evaluate-latest
```

`generate-next` 要求显式提供目标期号，不会把截止期号静默加一，也不会猜测跨年期号。正式生成默认拒绝 Git 脏工作区；测试或研究可显式传入 `--allow-dirty`，但制品仍会保存 `git_dirty=true` 和 `git_diff_hash`。历史少于默认 100 期时会拒绝运行；`--allow-short-history` 只用于测试和研究，并保存 `short_history_override=true`。

`NextPredictionArtifact` 保存目标期号、数据截止期号、带时区生成时间、Git commit SHA、Git 脏状态及差异哈希、历史文件 SHA-256、历史记录数与起止期号、展开后的完整 Pipeline 配置、全部随机种子、候选池摘要、最终 5 注和固定风险声明。外层目标期号、截止期号、生成时间、Pipeline 配置和随机种子必须与内部 `PredictionRecord` 完全一致，否则拒绝加载。`evaluate-latest` 同时接受纯 `PredictionRecord` JSON 和 `NextPredictionArtifact` JSON。

Pipeline 提供 `fast`、`standard`、`final` 三种运行 profile，实际候选数、搜索次数、稳定候选上限和候选评分并行数都会展开并保存在日志中；CLI 可用 `--candidate-count`、`--search-trials` 和 `--parallel-workers` 显式覆盖。CLI 已支持 `hot_cold_blend_score` 的 `--hot-window`、`--cold-window`、`--hot-weight` 和 `--decay`，这些参数由同一个 `ScorerSpec` 同时驱动执行和审计日志。默认回测只运行轻量随机 baseline；传入 `--include-optimized` 才逐历史期运行优化 Pipeline。

`reconcile_sources` 只报告冲突，不选择“更可信”的值；存在任何冲突时不返回可用于回测的合并表。数据质量 finding 中只有 `severity=error` 会令 `is_valid=false` 和 `blocks_backtest=true`；warning 会保留在报告中但不阻断回测。错误报告会使滚动回测立即停止，必须人工核实并重新提供一致数据。

## 历史规则与 ROI 边界

`PrizeRuleSchedule` 按 `effective_from_issue` 为每一期选择当时最近生效的规则。只有 2026 年七级规则配置时，早于第 26014 期的开奖不会套用该规则；命中指标仍会保留，但 `total_prize` 和 `roi` 为 `null`。

对于存在多个奖池上下文的规则，历史回测必须提供该期 `prize_context_by_issue` 或 `IssuePrizeRecord`；命中奖项但缺少上下文或浮动奖实际金额时，金额指标标为不可用。`IssuePrizeRecord` 绑定期号和规则版本，并允许用当期实际单注奖金覆盖规则配置。这样可以跨规则时期比较号码命中，但不会用 2026 年规则或猜测奖金计算旧期 ROI。

报告始终包含完全随机五注 baseline 和固定风险声明。当前框架不声称任何策略优于随机；在缺少
旧规则配置、逐期实际奖金、B1 配对结果、校准与最终留出结果前，不应做策略优越性结论。
