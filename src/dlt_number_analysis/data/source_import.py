"""Immutable source snapshots and explicit multi-source history verification."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Literal, Protocol, Self

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dlt_number_analysis.data.data_validator import (
    CSV_COLUMNS,
    DrawRecord,
    load_draws_csv,
    validate_draw_dataframe,
    write_validated_draws_csv,
)
from dlt_number_analysis.data.history_store import (
    SourceReconciliationResult,
    reconcile_sources,
)
from dlt_number_analysis.data.identity import canonical_history_sha256

SourceFormat = Literal["sporttery_json", "sporttery_json_pages", "five_hundred_html"]


class SourceSnapshot(BaseModel):
    """One immutable raw response plus sufficient provenance to audit it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_name: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    fetched_at: datetime
    source_format: SourceFormat
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    content: bytes = Field(min_length=1, repr=False)

    @field_validator("fetched_at")
    @classmethod
    def require_aware_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("fetched_at must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_content_hash(self) -> Self:
        if hashlib.sha256(self.content).hexdigest() != self.content_hash:
            raise ValueError("content_hash does not match snapshot content")
        return self

    @classmethod
    def from_content(
        cls,
        *,
        source_name: str,
        source_url: str,
        source_format: SourceFormat,
        content: bytes,
        fetched_at: datetime | None = None,
    ) -> SourceSnapshot:
        """Build a hash-bound snapshot from raw response bytes."""
        return cls(
            source_name=source_name,
            source_url=source_url,
            fetched_at=fetched_at or datetime.now(UTC),
            source_format=source_format,
            content_hash=hashlib.sha256(content).hexdigest(),
            content=content,
        )


class SourceFetcher(Protocol):
    """Network boundary that returns raw bytes without performing analysis."""

    def fetch(self, *, fetched_at: datetime | None = None) -> SourceSnapshot:
        """Fetch and return one immutable source response."""
        ...


class SourceSnapshotMetadata(BaseModel):
    """Serializable sidecar for a persisted raw snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_name: str
    source_url: str
    fetched_at: datetime
    source_format: SourceFormat
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_file: str


class ConflictResolutionRecord(BaseModel):
    """Human-authored, explicit selection for one reported conflicting issue."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    issue: str = Field(pattern=r"^\d+$")
    selected_source: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    resolved_by: str = Field(min_length=1)
    resolved_at: datetime
    source_content_hashes: dict[str, str] = Field(min_length=2)

    @field_validator("resolved_at")
    @classmethod
    def require_aware_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("resolved_at must include a timezone")
        return value


class VerifiedHistoryManifest(BaseModel):
    """Proof that one history file was reconciled from independent snapshots."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "verified-history-v1"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_names: tuple[str, ...] = Field(min_length=2)
    source_urls: tuple[str, ...] = Field(min_length=2)
    snapshot_content_hashes: tuple[str, ...] = Field(min_length=2)
    record_count: int = Field(ge=1)
    history_start_issue: str = Field(pattern=r"^\d+$")
    history_cutoff_issue: str = Field(pattern=r"^\d+$")
    canonical_history_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    conflict_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_independent_sources(self) -> Self:
        if len(set(self.source_names)) < 2:
            raise ValueError("verified history requires at least two independent sources")
        if len(self.source_names) != len(self.source_urls):
            raise ValueError("source names and URLs must align")
        if len(self.source_names) != len(self.snapshot_content_hashes):
            raise ValueError("source names and snapshot hashes must align")
        if self.conflict_count != 0:
            raise ValueError("verified history cannot contain unresolved conflicts")
        return self


class _HistoryTableParser(HTMLParser):
    """Extract table-cell text only from the 500.com history tbody."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._in_target_body = False
        self._in_row = False
        self._in_cell = False
        self._cell_parts: list[str] = []
        self._row: list[str] = []
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "tbody" and attributes.get("id") == "tdata":
            self._in_target_body = True
        elif self._in_target_body and tag == "tr":
            self._in_row = True
            self._row = []
        elif self._in_row and tag == "td":
            self._in_cell = True
            self._cell_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self._in_cell:
            self._row.append(" ".join("".join(self._cell_parts).split()))
            self._in_cell = False
        elif tag == "tr" and self._in_row:
            if self._row:
                self.rows.append(self._row)
            self._in_row = False
        elif tag == "tbody" and self._in_target_body:
            self._in_target_body = False


def write_source_snapshot(snapshot: SourceSnapshot, directory: str | Path) -> tuple[Path, Path]:
    """Persist raw bytes and metadata once; an existing target is never overwritten."""
    root = Path(directory) / snapshot.source_name
    root.mkdir(parents=True, exist_ok=True)
    timestamp = snapshot.fetched_at.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    if snapshot.source_format == "sporttery_json":
        suffix = ".json"
    elif snapshot.source_format == "sporttery_json_pages":
        suffix = ".pages.json"
    else:
        suffix = ".html"
    stem = f"{timestamp}_{snapshot.content_hash}"
    raw_path = root / f"{stem}{suffix}"
    metadata_path = root / f"{stem}.metadata.json"
    if raw_path.exists() or metadata_path.exists():
        raise FileExistsError("source snapshot already exists and cannot be overwritten")
    metadata = SourceSnapshotMetadata(
        source_name=snapshot.source_name,
        source_url=snapshot.source_url,
        fetched_at=snapshot.fetched_at,
        source_format=snapshot.source_format,
        content_hash=snapshot.content_hash,
        raw_file=raw_path.name,
    )
    raw_path.write_bytes(snapshot.content)
    metadata_path.write_text(metadata.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return raw_path, metadata_path


def load_source_snapshot(metadata_path: str | Path) -> SourceSnapshot:
    """Reconstruct and hash-check an immutable snapshot from its metadata sidecar."""
    source = Path(metadata_path)
    metadata = SourceSnapshotMetadata.model_validate_json(source.read_text(encoding="utf-8"))
    raw_path = source.parent / metadata.raw_file
    return SourceSnapshot(
        source_name=metadata.source_name,
        source_url=metadata.source_url,
        fetched_at=metadata.fetched_at,
        source_format=metadata.source_format,
        content_hash=metadata.content_hash,
        content=raw_path.read_bytes(),
    )


def _sporttery_payload_records(payload: object) -> list[object]:
    value = payload.get("value", payload) if isinstance(payload, dict) else payload
    records = value.get("list", []) if isinstance(value, dict) else value
    if not isinstance(records, list) or not records:
        raise ValueError("sporttery snapshot contains no draw records")
    return records


def _normalize_sporttery(snapshot: SourceSnapshot) -> list[dict[str, object]]:
    payload = json.loads(snapshot.content.decode("utf-8-sig"))
    if snapshot.source_format == "sporttery_json_pages":
        if not isinstance(payload, dict) or not isinstance(payload.get("pages"), list):
            raise ValueError("sporttery page bundle is invalid")
        import base64

        records: list[object] = []
        for page in payload["pages"]:
            if not isinstance(page, dict) or not isinstance(page.get("body_base64"), str):
                raise ValueError("sporttery page bundle entry is invalid")
            body = base64.b64decode(page["body_base64"], validate=True)
            if hashlib.sha256(body).hexdigest() != page.get("content_hash"):
                raise ValueError("sporttery bundled page hash does not match raw body")
            records.extend(_sporttery_payload_records(json.loads(body.decode("utf-8-sig"))))
    else:
        records = _sporttery_payload_records(payload)
    normalized: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("sporttery draw record must be an object")
        numbers = str(record["lotteryDrawResult"]).split()
        if len(numbers) != 7:
            raise ValueError("sporttery draw result must contain seven numbers")
        normalized.append(
            dict(
                zip(
                    CSV_COLUMNS,
                    [
                        str(record["lotteryDrawNum"]),
                        str(record["lotteryDrawTime"])[:10],
                        *(int(number) for number in numbers),
                    ],
                    strict=True,
                )
            )
        )
    return normalized


def _normalize_five_hundred(snapshot: SourceSnapshot) -> list[dict[str, object]]:
    parser = _HistoryTableParser()
    try:
        text = snapshot.content.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError:
        text = snapshot.content.decode("gb18030", errors="strict")
    parser.feed(text)
    normalized: list[dict[str, object]] = []
    for row in parser.rows:
        if len(row) < 15 or not row[0].isdigit():
            continue
        normalized.append(
            dict(
                zip(
                    CSV_COLUMNS,
                    [row[0], row[14], *(int(value) for value in row[1:8])],
                    strict=True,
                )
            )
        )
    if not normalized:
        raise ValueError("500.com snapshot contains no draw records")
    return normalized


def normalize_source_draws(snapshot: SourceSnapshot) -> pd.DataFrame:
    """Convert a supported raw snapshot into the shared validated history schema."""
    if snapshot.source_format in {"sporttery_json", "sporttery_json_pages"}:
        records = _normalize_sporttery(snapshot)
    else:
        records = _normalize_five_hundred(snapshot)
    records.sort(key=lambda item: int(str(item["issue"])))
    return validate_draw_dataframe(pd.DataFrame(records, columns=CSV_COLUMNS))


def reconcile_draw_sources(snapshots: Sequence[SourceSnapshot]) -> SourceReconciliationResult:
    """Normalize and compare at least two named sources without selecting conflicts."""
    if len(snapshots) < 2 or len({snapshot.source_name for snapshot in snapshots}) < 2:
        raise ValueError("at least two independent named source snapshots are required")
    if len({snapshot.source_name for snapshot in snapshots}) != len(snapshots):
        raise ValueError("one snapshot per source is required for reconciliation")
    return reconcile_sources(
        {snapshot.source_name: normalize_source_draws(snapshot) for snapshot in snapshots}
    )


def resolve_conflict_record(
    snapshots: Sequence[SourceSnapshot],
    resolution: ConflictResolutionRecord,
) -> DrawRecord:
    """Apply an explicit human resolution to one proven source conflict."""
    by_name = {snapshot.source_name: snapshot for snapshot in snapshots}
    if set(resolution.source_content_hashes) != set(by_name):
        raise ValueError("resolution must bind every reconciled source")
    for name, expected_hash in resolution.source_content_hashes.items():
        if by_name[name].content_hash != expected_hash:
            raise ValueError("resolution source hash does not match current snapshot")
    if resolution.selected_source not in by_name:
        raise ValueError("selected source is not present")
    reconciliation = reconcile_draw_sources(snapshots)
    conflict_issues = {
        finding.issue
        for finding in reconciliation.report.findings
        if finding.code == "source_conflict"
    }
    if resolution.issue not in conflict_issues:
        raise ValueError("resolution issue is not an unresolved source conflict")
    chosen = normalize_source_draws(by_name[resolution.selected_source])
    matching = chosen.loc[chosen["issue"].astype(str) == resolution.issue]
    if len(matching) != 1:
        raise ValueError("selected source does not contain exactly one resolution issue")
    return DrawRecord.model_validate(matching.iloc[0].to_dict())


def verified_manifest_path(history_path: str | Path) -> Path:
    """Return the deterministic sidecar path for a verified history file."""
    target = Path(history_path)
    return target.with_name(f"{target.name}.verified.json")


def write_verified_history(
    reconciliation: SourceReconciliationResult,
    snapshots: Sequence[SourceSnapshot],
    path: str | Path,
) -> VerifiedHistoryManifest:
    """Write history only after clean agreement between at least two independent sources."""
    if reconciliation.report.blocks_backtest or reconciliation.reconciled_draws is None:
        raise ValueError("unresolved source conflicts block verified history")
    source_names = tuple(sorted({snapshot.source_name for snapshot in snapshots}))
    if len(source_names) < 2:
        raise ValueError("verified history requires at least two independent sources")
    snapshots_by_name = {snapshot.source_name: snapshot for snapshot in snapshots}
    if set(source_names) != set(reconciliation.report.source_names):
        raise ValueError("snapshot sources do not match reconciliation report")
    draws = reconciliation.reconciled_draws
    manifest = VerifiedHistoryManifest(
        source_names=source_names,
        source_urls=tuple(snapshots_by_name[name].source_url for name in source_names),
        snapshot_content_hashes=tuple(
            snapshots_by_name[name].content_hash for name in source_names
        ),
        record_count=len(draws),
        history_start_issue=str(draws.iloc[0]["issue"]),
        history_cutoff_issue=str(draws.iloc[-1]["issue"]),
        canonical_history_sha256=canonical_history_sha256(draws),
        conflict_count=reconciliation.report.conflict_count,
    )
    target = write_validated_draws_csv(draws, path)
    sidecar = verified_manifest_path(target)
    sidecar.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest


def load_verified_history(path: str | Path) -> pd.DataFrame:
    """Load history only when its manifest still matches its logical contents."""
    source = Path(path)
    manifest_path = verified_manifest_path(source)
    if not manifest_path.exists():
        raise ValueError("formal operation requires a verified history manifest")
    manifest = VerifiedHistoryManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    draws = load_draws_csv(source)
    if len(draws) != manifest.record_count:
        raise ValueError("verified history record count no longer matches its manifest")
    if str(draws.iloc[0]["issue"]) != manifest.history_start_issue:
        raise ValueError("verified history start issue no longer matches its manifest")
    if str(draws.iloc[-1]["issue"]) != manifest.history_cutoff_issue:
        raise ValueError("verified history cutoff issue no longer matches its manifest")
    if canonical_history_sha256(draws) != manifest.canonical_history_sha256:
        raise ValueError("verified history was modified after reconciliation")
    return draws
