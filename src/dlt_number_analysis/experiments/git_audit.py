"""Git worktree path classification for resumable formal experiment execution."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


class GitStatusEntry(BaseModel):
    """One porcelain-v1 status entry with staged/worktree columns preserved."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    index_status: str
    worktree_status: str


class AuthorizedGeneratedPath(BaseModel):
    """One exact output file or directory derived from current CLI arguments."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    kind: Literal["file", "directory"]
    explicitly_provided: bool = False


class GitWorktreeAudit(BaseModel):
    """Distinguish complete cleanliness from formal-execution cleanliness."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    all_changed_paths: tuple[str, ...]
    authorized_generated_paths: tuple[str, ...]
    authorized_dirty_paths: tuple[str, ...]
    blocking_changed_paths: tuple[str, ...]
    tracked_source_changes: tuple[str, ...]
    untracked_source_changes: tuple[str, ...]
    staged_source_changes: tuple[str, ...]
    git_worktree_completely_clean: bool
    git_worktree_clean_for_formal_execution: bool
    authorized_dirty_path_count: int
    blocking_dirty_path_count: int
    blocking_dirty_paths: tuple[str, ...]


_PROTECTED_DIRECTORIES: tuple[str, ...] = (
    "src",
    "tests",
    "scripts",
    "config",
    "data/raw",
    ".github",
)
_PROTECTED_FILES: tuple[str, ...] = (
    "pyproject.toml",
    "uv.lock",
    ".gitignore",
)
_IMMUTABLE_GENERATED_DIRECTORIES: tuple[str, ...] = (
    "outputs/experiments/schema_v2",
    "outputs/experiments/schema_v3",
    "outputs/experiments/legacy_import",
    "outputs/experiments/final_holdout",
    "outputs/benchmarks/v055_identity_smoke",
    "outputs/benchmarks/v056_cohort_smoke",
)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _normalized_relative(path: Path, project_root: Path) -> str:
    try:
        value = path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        value = path.resolve().as_posix()
    return value


def _protected_paths(project_root: Path) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    directories = tuple((project_root / item).resolve() for item in _PROTECTED_DIRECTORIES)
    files = tuple((project_root / item).resolve() for item in _PROTECTED_FILES)
    return directories, files


def _is_protected_source(path: Path, project_root: Path) -> bool:
    directories, files = _protected_paths(project_root)
    resolved = path.resolve()
    return resolved in files or any(_is_within(resolved, directory) for directory in directories)


def _validate_authorized_paths(
    project_root: Path,
    authorized_paths: Sequence[AuthorizedGeneratedPath],
) -> tuple[AuthorizedGeneratedPath, ...]:
    root = project_root.resolve()
    protected_directories, protected_files = _protected_paths(root)
    immutable = tuple((root / item).resolve() for item in _IMMUTABLE_GENERATED_DIRECTORIES)
    validated: list[AuthorizedGeneratedPath] = []
    for item in authorized_paths:
        resolved = item.path.resolve()
        if resolved == root:
            raise ValueError("project root cannot be authorized as a generated output path")
        inside_project = _is_within(resolved, root)
        if not inside_project and not item.explicitly_provided:
            raise ValueError("external generated paths must be explicitly provided")
        forbidden_overlap = resolved in protected_files or any(
            _is_within(resolved, protected) or _is_within(protected, resolved)
            for protected in protected_directories
        )
        if forbidden_overlap:
            raise ValueError(f"source/config path cannot be authorized: {resolved}")
        if item.kind == "directory" and any(_is_within(prior, resolved) for prior in immutable):
            raise ValueError(
                "generated directory authorization cannot cover immutable prior-schema artifacts"
            )
        validated.append(item.model_copy(update={"path": resolved}))
    return tuple(validated)


def read_git_status_entries(project_root: str | Path) -> tuple[GitStatusEntry, ...]:
    """Read NUL-delimited porcelain status without losing spaces or rename paths."""
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=project_root,
        check=True,
        capture_output=True,
    )
    tokens = result.stdout.split(b"\0")
    entries: list[GitStatusEntry] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if not token:
            continue
        text = token.decode("utf-8", errors="surrogateescape")
        if len(text) < 4:
            raise ValueError("unexpected git porcelain entry")
        index_status, worktree_status = text[0], text[1]
        path = text[3:]
        if index_status in {"R", "C"} and index < len(tokens):
            renamed_to = tokens[index].decode("utf-8", errors="surrogateescape")
            index += 1
            path = renamed_to
        entries.append(
            GitStatusEntry(
                path=path,
                index_status=index_status,
                worktree_status=worktree_status,
            )
        )
    return tuple(entries)


def audit_git_worktree(
    project_root: str | Path,
    authorized_paths: Sequence[AuthorizedGeneratedPath],
    *,
    status_entries: Sequence[GitStatusEntry] | None = None,
) -> GitWorktreeAudit:
    """Classify dirty paths; authorization never overrides protected source changes."""
    root = Path(project_root).resolve()
    authorized = _validate_authorized_paths(root, authorized_paths)
    entries = tuple(status_entries) if status_entries is not None else read_git_status_entries(root)
    all_changed: list[str] = []
    authorized_dirty: list[str] = []
    blocking: list[str] = []
    tracked_source: list[str] = []
    untracked_source: list[str] = []
    staged_source: list[str] = []
    for entry in entries:
        absolute = (root / entry.path).resolve()
        relative = _normalized_relative(absolute, root)
        all_changed.append(relative)
        source = _is_protected_source(absolute, root)
        untracked = entry.index_status == "?" and entry.worktree_status == "?"
        staged = entry.index_status not in {" ", "?"}
        if source:
            if untracked:
                untracked_source.append(relative)
            else:
                tracked_source.append(relative)
            if staged:
                staged_source.append(relative)
        allowed = False
        if not source:
            for item in authorized:
                if item.kind == "file" and absolute == item.path:
                    allowed = True
                    break
                if item.kind == "directory" and _is_within(absolute, item.path):
                    allowed = True
                    break
        if allowed:
            authorized_dirty.append(relative)
        else:
            blocking.append(relative)
    all_values = tuple(sorted(set(all_changed)))
    authorized_values = tuple(sorted(set(authorized_dirty)))
    blocking_values = tuple(sorted(set(blocking)))
    return GitWorktreeAudit(
        all_changed_paths=all_values,
        authorized_generated_paths=tuple(
            sorted(_normalized_relative(item.path, root) for item in authorized)
        ),
        authorized_dirty_paths=authorized_values,
        blocking_changed_paths=blocking_values,
        tracked_source_changes=tuple(sorted(set(tracked_source))),
        untracked_source_changes=tuple(sorted(set(untracked_source))),
        staged_source_changes=tuple(sorted(set(staged_source))),
        git_worktree_completely_clean=not all_values,
        git_worktree_clean_for_formal_execution=not blocking_values,
        authorized_dirty_path_count=len(authorized_values),
        blocking_dirty_path_count=len(blocking_values),
        blocking_dirty_paths=blocking_values,
    )
