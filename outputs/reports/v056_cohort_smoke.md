# v0.5.6 logical cohort smoke

- targets: 08008, 10028, 12048, 14068, 16088, 18109
- initial tasks: 2
- initial task chunk sizes: [3, 3]
- cohort id: `v056_cohort_smoke_6`
- cohort definition SHA256: `ec201fd88465ceb2bfcb2c7e31d989b3b9fb1a830859ff6ea209d686e0e85c9e`
- expected targets SHA256: `97c3c374db59745eef546bcc8a07f1b7516a4b50131e29a164c6f22a9d62d86d`
- run context SHA256: `8c4a54ad4307985f7e5300cdc637e285bb940ff4d50a50d817f8bdfba3aae07a`
- observations by experiment: {'B1': 6, 'B2': 6, 'B3': 6, 'B4': 6, 'B5': 6, 'B6': 6}
- total observations: 36
- logical cohort count: 1
- paired target count: 6
- initial wrapper checkpoint schema collision repaired without rerunning computation: True
- resume validation: read_only_no_generation
- resume scheduled tasks: 0
- resume generation called: False
- report-only generation called: false
- schema version: `experiment-storage-schema-v3`

This bounded smoke validates identity, chunk independence, resume, and reporting boundaries only. It is not used for parameter selection and makes no strategy-advantage claim.

> 评分不等于真实中奖概率，彩票开奖结果是随机事件。
