# Pipeline 性能基准

peak_memory_mb 使用 tracemalloc，仅表示 Python 分配峰值，不包含解释器外全部内存。

| profile | parallel_workers | candidate_scoring_method | random_seed | runtime_seconds | peak_memory_mb | candidate_generation_seconds | candidate_scoring_seconds | portfolio_search_seconds | feasible_portfolios_evaluated | memory_method |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fast | 1 | numpy_batch_features | 20260714 | 11.0619 | 222.7079 | 0.5289 | 5.8584 | 1.9223 | 1332 | tracemalloc_python_allocations |
| fast | 4 | numpy_batch_features | 20260714 | 11.0374 | 222.4288 | 0.4516 | 6.0934 | 1.9686 | 1332 | tracemalloc_python_allocations |
| standard | 1 | numpy_batch_features | 20260714 | 18.0319 | 222.3193 | 0.4970 | 5.5820 | 9.5473 | 6610 | tracemalloc_python_allocations |
| standard | 4 | numpy_batch_features | 20260714 | 20.4397 | 222.4250 | 0.4917 | 6.7579 | 10.6255 | 6610 | tracemalloc_python_allocations |
| final | 1 | numpy_batch_features | 20260714 | 57.9992 | 556.4531 | 1.0726 | 14.5927 | 38.5101 | 26677 | tracemalloc_python_allocations |
| final | 4 | numpy_batch_features | 20260714 | 60.5510 | 556.5602 | 1.0990 | 15.3209 | 40.9469 | 26677 | tracemalloc_python_allocations |
| fast | 1 | scalar | 20260714 | 11.2564 | 222.7010 | 0.4181 | 6.7179 | 1.8771 | 1332 | tracemalloc_python_allocations |

NumPy 批量特征路径评估：fast/1-worker 候选评分加速比 1.147x（标量 / NumPy）；最终票据与种子配置相同。

建议 final 默认 parallel_workers=1。
只有 4 线程在每次配对测量中均达到稳定加速时，才会建议改为 4。

> 评分不等于真实中奖概率，彩票开奖结果是随机事件。
