# dlt-number-analysis

中国体育彩票超级大乐透（以下简称“大乐透”）历史开奖数据校验、统计特征、策略比较与严格时间序列滚动回测项目。

> **重要声明：评分不等于真实中奖概率，彩票开奖结果是随机事件。**

本项目只用于数据工程、统计分析与回测研究，不承诺、暗示或声称能够预测中奖号码。

## v0.2 当前范围

已实现：

- 历史开奖 CSV 的加载、结构校验、业务规则校验和校验后保存。
- 前后区静态数值特征、展示比例字符串及与上期重号数。
- 开奖来源、五注票据、预测日志、单注评估和复盘汇总 Pydantic 模型。
- 自第 26014 期生效的七级奖级 JSON 配置，以及完全配置驱动的评估函数。
- `max_coverage`、`core_rotation`、`hybrid_portfolio` 三种简单可替换策略。
- 每种策略固定五注、禁止完全重复，并保存模型版本、截止期号、生成时间、随机种子和全部参数。
- 严格扩展窗口滚动回测骨架及完全均匀随机五注 baseline。
- 第 26078 期用户提供的真实开奖、手工事前五注日志和复盘报告。

尚未实现：

- 自动数据抓取、官方来源交叉核验与增量更新。
- 完整历史开奖数据集和按期实际浮动奖金数据。
- 自动调度的下一期预测、开奖后评估和结果发布。
- 经充分样本验证的策略比较、参数选择和可视化报告。
- 复杂机器学习模型。

当前组合评分只是便于替换的工程占位实现，不代表真实中奖概率，也不声称任何策略优于随机。

## 环境与命令

要求 Python 3.12+，使用 [uv](https://docs.astral.sh/uv/) 管理依赖。

```powershell
uv sync --dev
uv run ruff format .
uv run ruff check .
uv run pytest
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
│   ├── backtests/           # 滚动回测结果
│   ├── predictions/         # 预测日志
│   └── reports/             # 复盘报告
├── scripts/                 # 可重复执行的制品生成脚本
├── src/dlt_number_analysis/
│   ├── data/                # CSV 结构和数据校验
│   ├── analysis/            # 前后区统计特征
│   ├── models/              # 来源、票据、预测和评估模型
│   ├── evaluation/          # 配置驱动的奖级评估与报告
│   ├── strategies/          # 五注组合策略及随机 baseline
│   ├── backtesting/         # 严格向前滚动回测
│   └── fetchers/            # 独立的数据抓取层（当前仅预留边界）
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
- 五注号码池覆盖全部 5 个前区和 2 个后区，但号码池全覆盖不代表单注预测成功。

可重复生成复盘：

```powershell
uv run python scripts/generate_26078_review.py
```

## 组合策略

三种策略均使用外部简单号码分数加带种子的随机平分排序，方便以后替换评分器：

- `max_coverage`：五注前区覆盖 25 个不同号码，后区覆盖 10 个不同号码，尽量减少跨票重复。
- `core_rotation`：五个核心前区号码各出现 2 或 3 次，其他位置优先低重复填充。
- `hybrid_portfolio`：3 注核心轮转与 2 注探索组合。
- `random_baseline`：完全均匀随机五注，不使用号码分数。

所有策略固定生成五注并由 `PredictionRecord` 禁止任意两注完全相同。策略输出只是候选组合记录，评分不等于中奖概率。

## 严格滚动回测

`run_rolling_backtest` 使用扩展窗口：预测索引为 `N` 的期开奖时，只用 `draws.iloc[:N]` 计算频率和生成票据，第 N 期实际开奖只在票据生成后用于评估。禁止随机切分历史时间序列。

逐期输出：

- `best_front_hits`
- `best_back_hits`
- `any_prize`
- `front_pool_coverage`
- `back_pool_coverage`
- `total_cost`
- `total_prize`
- `roi`，定义为 `(total_prize - total_cost) / total_cost`

报告始终包含完全随机五注 baseline 和固定风险声明。当前框架不声称任何策略优于随机；在缺少完整历史数据、实际浮动奖金和足够样本前，不应做策略优越性结论。
