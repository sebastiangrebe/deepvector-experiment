"""Shared utilities."""
from __future__ import annotations

import fnmatch
import logging
import sys
from pathlib import Path
from typing import Iterable

import torch


# ─────────────────────────────────────────────────────────────────────
# File filter (single source of truth for ALL retrievers)
# ─────────────────────────────────────────────────────────────────────

EXCLUDE_PATTERNS: list[str] = [
    "*/tests/*", "*/test/*", "*_test.py", "*_tests.py", "test_*.py",
    "*/docs/*", "*/doc/*",
    "*/build/*", "*/dist/*", "*/.git/*",
    "*/migrations/*",     # Django-specific noise
    "*/__pycache__/*",
]
INCLUDE_EXTENSIONS: list[str] = [".py"]
MAX_FILE_SIZE: int = 1_000_000  # 1 MB — matches published SWE-Bench retriever setups


def _matches_any(rel_path: str, patterns: Iterable[str]) -> bool:
    norm = "/" + rel_path.replace("\\", "/").lstrip("/")
    return any(fnmatch.fnmatch(norm, p if p.startswith("*") else "*" + p)
               or fnmatch.fnmatch(norm, p)
               for p in patterns)


def list_eligible_files(repo_path: Path) -> list[Path]:
    """Walk repo, return files passing the filter, sorted, repo-relative-stable."""
    out: list[Path] = []
    repo_path = repo_path.resolve()
    for ext in INCLUDE_EXTENSIONS:
        for p in repo_path.rglob(f"*{ext}"):
            if not p.is_file():
                continue
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if size > MAX_FILE_SIZE or size == 0:
                continue
            rel = p.relative_to(repo_path).as_posix()
            if _matches_any(rel, EXCLUDE_PATTERNS):
                continue
            out.append(p)
    out.sort()
    return out


def filter_stats(repo_path: Path) -> dict:
    """Return dict with counts: total, kept, excluded breakdown."""
    repo_path = repo_path.resolve()
    total = kept = too_big = empty = pattern_excluded = 0
    for ext in INCLUDE_EXTENSIONS:
        for p in repo_path.rglob(f"*{ext}"):
            if not p.is_file():
                continue
            total += 1
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if size == 0:
                empty += 1
                continue
            if size > MAX_FILE_SIZE:
                too_big += 1
                continue
            rel = p.relative_to(repo_path).as_posix()
            if _matches_any(rel, EXCLUDE_PATTERNS):
                pattern_excluded += 1
                continue
            kept += 1
    return {
        "total_py": total,
        "kept": kept,
        "excluded_pattern": pattern_excluded,
        "excluded_too_big": too_big,
        "excluded_empty": empty,
    }


# ─────────────────────────────────────────────────────────────────────
# Torch / logging / paths
# ─────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_logger(name: str = "deepvector") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent
