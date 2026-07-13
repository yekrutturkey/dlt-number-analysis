# dlt-number-analysis

中国体育彩票超级大乐透（以下简称“大乐透”）历史开奖数据校验、统计特征、策略比较与严格时间序列滚动回测项目。

> **重要声明：评分不等于真实中奖概率，彩票开奖结果是随机事件。**

本项目只用于数据工程、统计分析与回测研究，不承诺、暗示或声称能够预测中奖号码。

## 当前范围

已实现：

- 历史开奖 CSV 的加载、结构校验、业务规则校验和校验后保存。
- 前区和值、跨度、奇偶比、大小比、三区比、连号对数、同尾对数和与上期重号数。
- 数据校验与特征工程的单元测试。

尚未实现：

- 数据源抓取与增量更新。
- 候选组合生成、评分和下一期 5 注候选号码。
- 每期开奖后对上一期预测进行评估。
- 严格的时间序列滚动回测和策略比较。
- 图表与分析报告输出。

暂不引入复杂机器学习模型。

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
├── outputs/
│   ├── backtests/           # 未来的滚动回测结果
│   └── predictions/         # 未来的预测记录
├── src/dlt_number_analysis/
│   ├── data/                # CSV 结构和数据校验
│   ├── analysis/            # 统计特征与后续分析
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

`engineer_features` 会先校验输入，再按 CSV 中的时间顺序计算以下前区特征：

| 输出列 | 定义 |
| --- | --- |
| `front_sum` | 5 个前区号码之和 |
| `front_span` | 最大前区号码减最小前区号码 |
| `odd_even_ratio` | 奇数个数与偶数个数，格式为 `奇:偶` |
| `large_small_ratio` | 大号（18–35）与小号（1–17）个数，格式为 `大:小` |
| `zone_ratio` | 一区（1–12）、二区（13–24）、三区（25–35）个数，格式为 `一区:二区:三区` |
| `consecutive_pair_count` | 排序后差为 1 的相邻号码对数；例如 `1,2,3` 计 2 对 |
| `same_tail_pair_count` | 个位数相同的号码对数；例如 `1,11,21` 计 3 对 |
| `repeat_from_previous_count` | 与上一期前区号码的交集大小；第一期为缺失值 |

```python
from dlt_number_analysis.analysis import engineer_features

featured = engineer_features(draws)
```

第一期的重号数使用 `pandas.NA`，因为“没有上期”与“有上期但重号数为 0”不是同一含义。只允许使用当前期及之前的数据计算某一期特征，不得使用未来开奖数据生成历史预测。

## 后续预测与回测约束

未来每次预测记录必须包含：

- 模型/策略版本；
- 数据截止期号；
- 带时区的生成时间；
- 完整生成参数；
- 候选号码及其评分；
- 固定声明：**评分不等于真实中奖概率，彩票开奖结果是随机事件。**

历史预测只能使用当期预测时点已经可获得的数据。回测必须按时间向前滚动，在每个切分点重新拟合或统计，禁止随机拆分历史数据和任何形式的未来数据泄漏。
