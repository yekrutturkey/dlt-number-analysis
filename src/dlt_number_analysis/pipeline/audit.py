"""Reproducible history and Git audit metadata for prediction artifacts."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class HistoryAudit(BaseModel):
    """Identity and temporal bounds of the exact history file used by the CLI."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    record_count: int = Field(ge=1)
    start_issue: str = Field(pattern=r"^\d+$")
    cutoff_issue: str = Field(pattern=r"^\d+$")


class GitAudit(BaseModel):
    """Commit and working-tree identity captured before a formal generation run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    commit_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    dirty: bool
    diff_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


def build_history_audit(path: str | Path, draws: pd.DataFrame) -> HistoryAudit:
    """Hash the exact CSV bytes and bind them to the loaded history bounds."""
    source = Path(path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return HistoryAudit(
        sha256=digest,
        record_count=len(draws),
        start_issue=str(draws.iloc[0]["issue"]),
        cutoff_issue=str(draws.iloc[-1]["issue"]),
    )


def _run_git_bytes(project_root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=project_root,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def collect_git_audit(project_root: str | Path) -> GitAudit:
    """Hash tracked diffs plus porcelain status, including untracked path names."""
    root = Path(project_root)
    commit_sha = _run_git_bytes(root, "rev-parse", "HEAD").decode("ascii").strip()
    status = _run_git_bytes(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    diff = _run_git_bytes(root, "diff", "--binary", "HEAD", "--")
    digest = hashlib.sha256()
    digest.update(status)
    digest.update(b"\0")
    digest.update(diff)
    return GitAudit(
        commit_sha=commit_sha,
        dirty=bool(status.strip()),
        diff_hash=digest.hexdigest(),
    )
