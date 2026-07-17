"""Formal report isolation, read-only preflight, and report-only command tests."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from dlt_number_analysis.data import (
    CSV_COLUMNS,
    VerifiedHistoryManifest,
    canonical_history_sha256,
    verified_manifest_path,
)
from dlt_number_analysis.experiments import command
from dlt_number_analysis.experiments.cohort import build_cohort_definition_identity
from dlt_number_analysis.experiments.git_audit import (
    AuthorizedGeneratedPath,
    GitStatusEntry,
)
from dlt_number_analysis.experiments.identity import (
    build_experiment_execution_identity,
    build_run_context_identity,
)
from dlt_number_analysis.experiments.manual_operations import ManualOperationsAuditReport
from dlt_number_analysis.experiments.preflight import (
    build_experiment_preflight,
    write_experiment_preflight,
)
from dlt_number_analysis.experiments.reporting import (
    assign_observation_cohorts,
    compare_to_constraint_matched_baseline,
    summarize_observations,
    validate_single_formal_report_context,
)
from dlt_number_analysis.experiments.specs import baseline_experiment_specs
from dlt_number_analysis.experiments.storage import STORAGE_SCHEMA_VERSION
from dlt_number_analysis.experiments.storage_v4 import (
    ExperimentPartitionStatusV4,
    PartitionGenerationAudit,
)


def _history(count: int = 20) -> pd.DataFrame:
    start = date(2020, 1, 1)
    return pd.DataFrame(
        [
            [
                str(20001 + index),
                (start + timedelta(days=index * 2)).isoformat(),
                1,
                2,
                3,
                4,
                5 + index % 15,
                1,
                2 + index % 10,
            ]
            for index in range(count)
        ],
        columns=CSV_COLUMNS,
    )


def _report_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for experiment_index, experiment_id in enumerate(("B1", "B2", "B3", "B4", "B5", "B6")):
        for target_index, target in enumerate(("100", "101", "102", "103", "104", "105")):
            rows.append(
                {
                    "phase": "development",
                    "cohort_id": "v056_cohort_smoke_6",
                    "cohort_definition_sha256": "a" * 64,
                    "run_context_sha256": "b" * 64,
                    "history_sha256": "c" * 64,
                    "evaluation_mode": "raw_observation",
                    "profile": "fast",
                    "storage_schema_version": STORAGE_SCHEMA_VERSION,
                    "formal_inference_eligible": True,
                    "identity_status": "formal_verified",
                    "experiment_id": experiment_id,
                    "experiment_version": f"v-{experiment_id}",
                    "experiment_config_sha256": f"{experiment_index + 1:x}" * 64,
                    "execution_config_sha256": f"{experiment_index + 7:x}" * 64,
                    "execution_identity_schema_version": "v2",
                    "seed": 20260000,
                    "target_issue": target,
                    "task_id": "chunk-0" if target_index < 3 else "chunk-1",
                    "task_index": 0 if target_index < 3 else 1,
                    "task_targets_sha256": ("d" if target_index < 3 else "e") * 64,
                    "draw_date": f"2020-01-{target_index + 1:02d}",
                    "best_front_hits": 1 + (experiment_index > 0),
                    "best_back_hits": 0,
                    "best_total_hits": 1 + (experiment_index > 0),
                    "at_least_three_front": False,
                    "at_least_2_plus_1": False,
                    "unique_hit_concentration": 0.5,
                    "any_prize": False,
                    "total_prize": 0.0,
                    "roi": -1.0,
                }
            )
    return pd.DataFrame(rows)


def _preflight_inputs(tmp_path: Path):
    draws = _history()
    history_path = tmp_path / "draws.csv"
    history_path.write_text("unused", encoding="utf-8")
    manifest = VerifiedHistoryManifest(
        created_at=datetime.now(UTC),
        source_names=("one", "two"),
        source_urls=("https://one.invalid", "https://two.invalid"),
        snapshot_content_hashes=("1" * 64, "2" * 64),
        record_count=len(draws),
        history_start_issue=str(draws.iloc[0]["issue"]),
        history_cutoff_issue=str(draws.iloc[-1]["issue"]),
        canonical_history_sha256=canonical_history_sha256(draws),
        conflict_count=0,
    )
    verified_manifest_path(history_path).write_text(manifest.model_dump_json(), encoding="utf-8")
    specs = baseline_experiment_specs(seeds=(77,), phase="development")[1:3]
    context = build_run_context_identity(
        draws,
        data_split_spec=specs[0].data_split,
        phase="development",
        evaluation_mode="raw_observation",
        profile="fast",
        portfolio_scoring_method="numpy_vectorized",
        minimum_history=3,
        minimum_bank_size=500,
        maximum_bank_search_trials=80_000,
    )
    identities = {
        spec.experiment_id: build_experiment_execution_identity(context, spec) for spec in specs
    }
    cohort = build_cohort_definition_identity(
        ("20005", "20006"),
        cohort_purpose="development_smoke",
        phase="development",
        cohort_id="preflight",
        phase_target_issues=tuple(draws.iloc[:12]["issue"].astype(str)),
    )
    statuses = {
        (spec.experiment_id, 77): ExperimentPartitionStatusV4(
            phase="development",
            experiment_id=spec.experiment_id,
            experiment_version=spec.experiment_version,
            run_context_sha256=context.run_context_sha256,
            execution_config_sha256=identities[spec.experiment_id].execution_config_sha256,
            cohort_definition_sha256=cohort.cohort_definition_sha256,
            expected_targets_sha256=cohort.payload.expected_targets_sha256,
            seed=77,
            current_path=tmp_path / spec.experiment_id / "CURRENT",
            current_generation_id=None,
            completed_target_issues=(),
            pending_target_issues=("20005", "20006"),
            is_complete=False,
        )
        for spec in specs
    }
    return draws, history_path, specs, context, identities, cohort, statuses


def test_two_task_chunks_aggregate_as_one_logical_cohort() -> None:
    summary = summarize_observations(_report_rows())
    assert summary["cohort_definition_sha256"].nunique() == 1
    assert set(summary["observation_count"].astype(int)) == {6}


def test_paired_comparison_merges_targets_across_task_chunks() -> None:
    comparisons = compare_to_constraint_matched_baseline(
        _report_rows(),
        bootstrap_resamples=100,
        permutations=100,
        minimum_paired_observations=2,
    )
    assert comparisons
    assert {comparison.common_target_count for comparison in comparisons} == {6}


@pytest.mark.parametrize(
    ("column", "value"),
    (
        ("cohort_definition_sha256", "f" * 64),
        ("run_context_sha256", "f" * 64),
        ("evaluation_mode", "full_resampling"),
    ),
)
def test_mixed_formal_context_is_rejected(column: str, value: str) -> None:
    rows = _report_rows()
    rows.loc[0, column] = value
    with pytest.raises(ValueError, match="mixes"):
        validate_single_formal_report_context(rows)


def test_schema_v3_missing_explicit_cohort_identity_is_rejected() -> None:
    rows = _report_rows()
    rows.loc[:, "cohort_id"] = ""
    with pytest.raises(ValueError, match="inference is forbidden"):
        assign_observation_cohorts(rows)


def test_legacy_cohort_inference_is_marked_informal() -> None:
    row = (
        _report_rows()
        .iloc[[0]]
        .drop(columns=["storage_schema_version", "cohort_definition_sha256"])
    )
    row.loc[:, "cohort_id"] = ""
    row.loc[:, "target_issue"] = "8009"
    annotated = assign_observation_cohorts(row)
    assert annotated.iloc[0]["cohort_identity_source"] == "legacy_inferred"
    assert not bool(annotated.iloc[0]["formal_inference_eligible"])


def test_preflight_v2_checks_do_not_generate(tmp_path: Path) -> None:
    inputs = _preflight_inputs(tmp_path)
    plan = build_experiment_preflight(
        project_root=tmp_path,
        history_path=inputs[1],
        draws=inputs[0],
        specifications=inputs[2],
        run_context=inputs[3],
        identities=inputs[4],
        cohort=inputs[5],
        statuses=inputs[6],
        task_chunk_plan=({"task_id": "one"},),
        result_output_paths=(tmp_path / "schema_v4",),
        inference_context="formal",
        allow_dirty=False,
        git_status_entries=(),
        results_root=tmp_path / "schema_v4",
    )
    assert len(plan.checks) >= 30
    assert plan.schema_version == "experiment-preflight-v2"
    assert plan.ready


def test_formal_preflight_rejects_dirty_worktree(tmp_path: Path) -> None:
    inputs = _preflight_inputs(tmp_path)
    plan = build_experiment_preflight(
        project_root=tmp_path,
        history_path=inputs[1],
        draws=inputs[0],
        specifications=inputs[2],
        run_context=inputs[3],
        identities=inputs[4],
        cohort=inputs[5],
        statuses=inputs[6],
        task_chunk_plan=(),
        result_output_paths=(tmp_path / "schema_v4",),
        inference_context="formal",
        allow_dirty=False,
        git_status_entries=(
            GitStatusEntry(path="src/changed.py", index_status=" ", worktree_status="M"),
        ),
        results_root=tmp_path / "schema_v4",
    )
    assert not plan.ready
    assert plan.execution_will_be_blocked
    assert plan.estimated_task_count == 0
    assert plan.blocking_dirty_paths == ("src/changed.py",)


def test_smoke_preflight_allows_explicit_dirty_override(tmp_path: Path) -> None:
    inputs = _preflight_inputs(tmp_path)
    generated = tmp_path / "outputs/formal_runs/run/log.txt"
    plan = build_experiment_preflight(
        project_root=tmp_path,
        history_path=inputs[1],
        draws=inputs[0],
        specifications=inputs[2],
        run_context=inputs[3],
        identities=inputs[4],
        cohort=inputs[5],
        statuses=inputs[6],
        task_chunk_plan=(),
        result_output_paths=(tmp_path / "schema_v4",),
        inference_context="smoke",
        allow_dirty=True,
        authorized_generated_paths=(
            AuthorizedGeneratedPath(
                path=tmp_path / "outputs/formal_runs/run",
                kind="directory",
            ),
        ),
        git_status_entries=(
            GitStatusEntry(
                path=generated.relative_to(tmp_path).as_posix(),
                index_status="?",
                worktree_status="?",
            ),
        ),
        results_root=tmp_path / "schema_v4",
    )
    assert plan.ready and plan.git_dirty


def test_preflight_outputs_use_exact_identity_directories(tmp_path: Path) -> None:
    inputs = _preflight_inputs(tmp_path)
    plan = build_experiment_preflight(
        project_root=tmp_path,
        history_path=inputs[1],
        draws=inputs[0],
        specifications=inputs[2],
        run_context=inputs[3],
        identities=inputs[4],
        cohort=inputs[5],
        statuses=inputs[6],
        task_chunk_plan=(),
        result_output_paths=(tmp_path / "schema_v4",),
        inference_context="formal",
        allow_dirty=False,
        git_status_entries=(),
        results_root=tmp_path / "schema_v4",
    )
    json_path, markdown_path = write_experiment_preflight(plan, tmp_path / "plans")
    assert plan.run_context_sha256 in json_path.parts
    assert plan.cohort_definition_sha256 in markdown_path.parts


def test_preflight_truthfully_reports_generation_and_identity_conflicts(
    tmp_path: Path,
) -> None:
    inputs = _preflight_inputs(tmp_path)
    audit = PartitionGenerationAudit(
        partition_directory=tmp_path / "schema_v4/partition",
        current_generation_id="broken",
        valid_generation_ids=("valid-orphan",),
        orphan_generation_ids=("valid-orphan",),
        invalid_generation_ids=("broken",),
        current_is_valid=False,
        recovery_possible=True,
        warnings=("CURRENT invalid",),
    )
    plan = build_experiment_preflight(
        project_root=tmp_path,
        history_path=inputs[1],
        draws=inputs[0],
        specifications=inputs[2],
        run_context=inputs[3],
        identities=inputs[4],
        cohort=inputs[5],
        statuses=inputs[6],
        task_chunk_plan=(),
        result_output_paths=(tmp_path / "schema_v4",),
        inference_context="formal",
        allow_dirty=False,
        git_status_entries=(),
        generation_audits={"B1:seed_77": audit},
        results_root=tmp_path / "schema_v4",
        identity_conflicts=("expected target hash conflict",),
    )
    assert not plan.ready
    assert plan.orphan_generation_count == 1
    assert plan.invalid_generation_count == 1
    assert "expected target hash conflict" in plan.identity_conflicts
    assert plan.current_generation_by_partition == {"B1:seed_77": "broken"}


def test_normal_execution_stops_before_task_construction_when_preflight_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draws = _history()
    generation_called = False
    task_construction_called = False

    class FakeStore:
        def __init__(self, _: Path) -> None:
            pass

        def current_path(self, spec, identity, cohort, *, seed: int) -> Path:
            return tmp_path / "schema_v4" / spec.experiment_id / f"seed_{seed}" / "CURRENT"

        def audit_partition_generations(self, spec, identity, cohort, *, seed: int):
            return PartitionGenerationAudit(
                partition_directory=tmp_path / "schema_v4" / spec.experiment_id,
                current_generation_id=None,
                valid_generation_ids=(),
                orphan_generation_ids=(),
                invalid_generation_ids=(),
                current_is_valid=True,
                recovery_possible=False,
                warnings=(),
            )

        def status(
            self,
            spec,
            identity,
            run_context,
            cohort,
            *,
            seed: int,
            expected_target_issues,
        ) -> ExperimentPartitionStatusV4:
            targets = tuple(expected_target_issues)
            return ExperimentPartitionStatusV4(
                phase=spec.data_split.phase,
                experiment_id=spec.experiment_id,
                experiment_version=spec.experiment_version,
                run_context_sha256=identity.run_context_sha256,
                execution_config_sha256=identity.execution_config_sha256,
                cohort_definition_sha256=cohort.cohort_definition_sha256,
                expected_targets_sha256=cohort.payload.expected_targets_sha256,
                seed=seed,
                current_path=self.current_path(spec, identity, cohort, seed=seed),
                current_generation_id=None,
                completed_target_issues=(),
                pending_target_issues=targets,
                is_complete=False,
            )

        def append(self, _: pd.DataFrame) -> None:
            nonlocal generation_called
            generation_called = True

    def forbidden_task_construction(*args, **kwargs):
        nonlocal task_construction_called
        task_construction_called = True
        pytest.fail("preflight failure constructed scheduler tasks")

    marker = tmp_path / "preflight-written.txt"
    monkeypatch.setattr(command, "ExperimentResultStoreV4", FakeStore)
    monkeypatch.setattr(command, "load_verified_history", lambda _: draws)
    monkeypatch.setattr(
        command,
        "build_experiment_preflight",
        lambda **_: SimpleNamespace(ready=False),
    )
    monkeypatch.setattr(
        command,
        "write_experiment_preflight",
        lambda *_: marker.write_text("blocked", encoding="utf-8"),
    )
    monkeypatch.setattr(command, "build_process_tasks", forbidden_task_construction)
    monkeypatch.setattr(
        command,
        "FormalRunLock",
        lambda *_args, **_kwargs: pytest.fail("preflight failure acquired run lock"),
    )
    results_root = tmp_path / "schema_v4"
    with pytest.raises(RuntimeError, match="execution was not started"):
        command.main(
            [
                "--experiment-ids",
                "B1",
                "--target-issues",
                "20005",
                "--minimum-history",
                "3",
                "--results-root",
                str(results_root),
                "--preflight-root",
                str(tmp_path / "plans"),
            ]
        )
    assert marker.exists()
    assert not generation_called
    assert not task_construction_called
    assert not results_root.exists()


def test_manual_operations_audit_mode_is_read_only_and_does_not_lock_or_generate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    draws = _history()
    audit_called = False

    class FakeStore:
        def __init__(self, _: Path) -> None:
            pass

        def current_path(self, spec, identity, cohort, *, seed: int) -> Path:
            return tmp_path / "schema_v4" / spec.experiment_id / f"seed_{seed}" / "CURRENT"

        def audit_partition_generations(self, spec, identity, cohort, *, seed: int):
            return PartitionGenerationAudit(
                partition_directory=tmp_path / "schema_v4" / spec.experiment_id,
                current_generation_id=None,
                valid_generation_ids=(),
                orphan_generation_ids=(),
                invalid_generation_ids=(),
                current_is_valid=True,
                recovery_possible=False,
                warnings=(),
            )

        def status(
            self,
            spec,
            identity,
            run_context,
            cohort,
            *,
            seed: int,
            expected_target_issues,
        ) -> ExperimentPartitionStatusV4:
            targets = tuple(expected_target_issues)
            return ExperimentPartitionStatusV4(
                phase=spec.data_split.phase,
                experiment_id=spec.experiment_id,
                experiment_version=spec.experiment_version,
                run_context_sha256=identity.run_context_sha256,
                execution_config_sha256=identity.execution_config_sha256,
                cohort_definition_sha256=cohort.cohort_definition_sha256,
                expected_targets_sha256=cohort.payload.expected_targets_sha256,
                seed=seed,
                current_path=self.current_path(spec, identity, cohort, seed=seed),
                current_generation_id=None,
                completed_target_issues=(),
                pending_target_issues=targets,
                is_complete=False,
            )

    def fake_manual_audit(_):
        nonlocal audit_called
        audit_called = True
        return ManualOperationsAuditReport(
            prepared_operations=(),
            completed_operations=(),
            failed_operations=(),
            inconsistent_operations=(),
            warnings=(),
        )

    monkeypatch.setattr(command, "ExperimentResultStoreV4", FakeStore)
    monkeypatch.setattr(command, "load_verified_history", lambda _: draws)
    monkeypatch.setattr(command, "audit_manual_operations", fake_manual_audit)
    monkeypatch.setattr(
        command,
        "FormalRunLock",
        lambda *_args, **_kwargs: pytest.fail("manual audit acquired run lock"),
    )
    monkeypatch.setattr(
        command,
        "build_process_tasks",
        lambda *_args, **_kwargs: pytest.fail("manual audit constructed tasks"),
    )
    monkeypatch.setattr(
        command,
        "build_experiment_preflight",
        lambda **_: pytest.fail("manual audit entered formal preflight execution"),
    )
    results_root = tmp_path / "schema_v4"
    result = command.main(
        [
            "--audit-manual-operations",
            "--experiment-ids",
            "B1",
            "--target-issues",
            "20005",
            "--minimum-history",
            "3",
            "--results-root",
            str(results_root),
        ]
    )
    output = capsys.readouterr().out
    assert result == 0
    assert audit_called
    assert "prepared_operations: 0" in output
    assert "inconsistent_operations: 0" in output
    assert not results_root.exists()


def test_report_only_does_not_invoke_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _report_rows().loc[lambda frame: frame["experiment_id"] == "B1"].copy()

    class FakeStore:
        def __init__(self, _: Path) -> None:
            pass

        def load_cohort(self, **_: object) -> pd.DataFrame:
            return rows

    monkeypatch.setattr(command, "ExperimentResultStoreV4", FakeStore)
    monkeypatch.setattr(
        command,
        "execute_experiment_process_task",
        lambda _: pytest.fail("report-only invoked generation"),
    )
    monkeypatch.setattr(
        command,
        "FormalRunLock",
        lambda *_args, **_kwargs: pytest.fail("report-only acquired run lock"),
    )
    result = command.main(
        [
            "--report-only",
            "--experiment-ids",
            "B1",
            "--run-context-sha256",
            "b" * 64,
            "--cohort-definition-sha256",
            "a" * 64,
            "--summary-output",
            str(tmp_path / "report.md"),
            "--inference-context",
            "smoke",
        ]
    )
    assert result == 0
    assert (tmp_path / "report.md").exists()
