# v0.5.5 execution identity smoke

- status: completed
- targets: 08008, 13058, 18109
- observation_count: 18
- run_seconds: 17.665986299994984
- target_seconds_range: {'maximum': 6.701776300003985, 'median': 6.17567599999893, 'minimum': 3.8417468999978155}
- run_context_sha256: `36188942f1826fd5578c97399b3c7a8023681bbe85721a015b266a9db3817927`
- history_sha256: `4481e6d5354bf277adf0186d184a357f64f266e9846fc0d2f81711d0f36f1128`
- old_results_did_not_skip: True
- duplicate_primary_key_rejected: True
- manifest_tamper_rejected: True
- resume_validation: {"resume_validation": "read_only_no_generation", "targets": ["08008", "13058", "18109"], "completed_partition_count": 6, "observation_count": 18, "generation_called": false, "risk_disclaimer": "评分不等于真实中奖概率，彩票开奖结果是随机事件。"}

This bounded smoke validates execution identity and resumability only. It is not used for parameter selection and does not establish a strategy advantage.

> 评分不等于真实中奖概率，彩票开奖结果是随机事件。
