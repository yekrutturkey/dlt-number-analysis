# Experiment preflight

- Ready: `true`
- Phase: `development`
- Cohort: `v056_cohort_smoke_6`
- Run context: `8c4a54ad4307985f7e5300cdc637e285bb940ff4d50a50d817f8bdfba3aae07a`
- Cohort definition: `ec201fd88465ceb2bfcb2c7e31d989b3b9fb1a830859ff6ea209d686e0e85c9e`
- Expected targets: `6`
- Estimated tasks: `2`

## Checks

- FAIL `git_worktree_clean`: False
- PASS `git_commit_sha`: df52ebd4ef07ffa353a5ff89ff7061c3f7c7e564
- PASS `verified_history`: verified-history-v1
- PASS `canonical_history_sha256`: 4481e6d5354bf277adf0186d184a357f64f266e9846fc0d2f81711d0f36f1128
- PASS `phase_and_split_boundaries`: development
- PASS `experiment_versions`: B1:v0.5.5-b1-shared-bank-v2,B2:v0.5.5-b2-shared-bank-v2,B3:v0.5.5-b3-shared-bank-v2,B4:v0.5.5-b4-shared-bank-v2,B5:v0.5.5-b5-shared-bank-v2,B6:v0.5.5-b6-shared-bank-v2
- PASS `evaluation_mode`: raw_observation
- PASS `profile`: fast
- PASS `portfolio_scoring_method`: numpy_vectorized
- PASS `run_context_sha256`: 8c4a54ad4307985f7e5300cdc637e285bb940ff4d50a50d817f8bdfba3aae07a
- PASS `execution_config_sha256`: 6
- PASS `cohort_id`: v056_cohort_smoke_6
- PASS `cohort_definition_sha256`: ec201fd88465ceb2bfcb2c7e31d989b3b9fb1a830859ff6ea209d686e0e85c9e
- PASS `expected_targets_sha256`: 97c3c374db59745eef546bcc8a07f1b7516a4b50131e29a164c6f22a9d62d86d
- PASS `expected_target_count`: 6
- PASS `cohort_start_end`: 08008-18109
- PASS `completed_target_counts`: {'B1:seed_20260000': 0, 'B2:seed_20260000': 0, 'B3:seed_20260000': 0, 'B4:seed_20260000': 0, 'B5:seed_20260000': 0, 'B6:seed_20260000': 0}
- PASS `pending_target_counts`: {'B1:seed_20260000': 6, 'B2:seed_20260000': 6, 'B3:seed_20260000': 6, 'B4:seed_20260000': 6, 'B5:seed_20260000': 6, 'B6:seed_20260000': 6}
- PASS `task_chunk_plan`: 2
- PASS `result_output_paths`: E:\dlt-number-analysis\outputs\benchmarks\v056_cohort_smoke\schema_v3\development\B1\v0.5.5-b1-shared-bank-v2\8c4a54ad4307985f7e5300cdc637e285bb940ff4d50a50d817f8bdfba3aae07a\b1d22089dac7b8b4a4dfe2e7dece437157b4a87fecfd169822fbc2730fca8a16\ec201fd88465ceb2bfcb2c7e31d989b3b9fb1a830859ff6ea209d686e0e85c9e\seed_20260000\observations.parquet;E:\dlt-number-analysis\outputs\benchmarks\v056_cohort_smoke\schema_v3\development\B2\v0.5.5-b2-shared-bank-v2\8c4a54ad4307985f7e5300cdc637e285bb940ff4d50a50d817f8bdfba3aae07a\71098076a4c94f70c98d786cc2749c956418ca04d4e45fbfb874c67cd652e521\ec201fd88465ceb2bfcb2c7e31d989b3b9fb1a830859ff6ea209d686e0e85c9e\seed_20260000\observations.parquet;E:\dlt-number-analysis\outputs\benchmarks\v056_cohort_smoke\schema_v3\development\B3\v0.5.5-b3-shared-bank-v2\8c4a54ad4307985f7e5300cdc637e285bb940ff4d50a50d817f8bdfba3aae07a\605bd1e5f2f06a277917e125b1d11f072b142c46839e23e6832f1e0fb81c89b9\ec201fd88465ceb2bfcb2c7e31d989b3b9fb1a830859ff6ea209d686e0e85c9e\seed_20260000\observations.parquet;E:\dlt-number-analysis\outputs\benchmarks\v056_cohort_smoke\schema_v3\development\B4\v0.5.5-b4-shared-bank-v2\8c4a54ad4307985f7e5300cdc637e285bb940ff4d50a50d817f8bdfba3aae07a\4778edcceab26f1a96422906352a14accd384501a91a64a66358114bff9d3110\ec201fd88465ceb2bfcb2c7e31d989b3b9fb1a830859ff6ea209d686e0e85c9e\seed_20260000\observations.parquet;E:\dlt-number-analysis\outputs\benchmarks\v056_cohort_smoke\schema_v3\development\B5\v0.5.5-b5-shared-bank-v2\8c4a54ad4307985f7e5300cdc637e285bb940ff4d50a50d817f8bdfba3aae07a\d843806b2d9d305d2cc4c2ce251cc6b41ce37bb631463ca0b5948cededc57fd6\ec201fd88465ceb2bfcb2c7e31d989b3b9fb1a830859ff6ea209d686e0e85c9e\seed_20260000\observations.parquet;E:\dlt-number-analysis\outputs\benchmarks\v056_cohort_smoke\schema_v3\development\B6\v0.5.5-b6-shared-bank-v2\8c4a54ad4307985f7e5300cdc637e285bb940ff4d50a50d817f8bdfba3aae07a\3ae8d39408e37a3d34985175ee3d39dcc27456c1c76eb74dfcc0ea1de8c96b80\ec201fd88465ceb2bfcb2c7e31d989b3b9fb1a830859ff6ea209d686e0e85c9e\seed_20260000\observations.parquet
- PASS `identity_conflicts`: none
- PASS `touches_legacy_or_schema_v2`: false
- PASS `reads_final_holdout_results`: false
- PASS `estimated_task_count`: 2

评分不等于真实中奖概率，彩票开奖结果是随机事件。
