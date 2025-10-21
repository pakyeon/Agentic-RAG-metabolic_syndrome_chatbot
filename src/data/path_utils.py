"""Utility helpers for project-relative paths."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

# Project root: two levels up from this file (src/data/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Default data locations within the repository
DEFAULT_PERSIST_DIRECTORY = PROJECT_ROOT / "chromadb" / "openai"
DEFAULT_PARSED_DIRECTORY = PROJECT_ROOT / "metabolic_syndrome_data" / "parsed"
DEFAULT_RAW_DIRECTORY = PROJECT_ROOT / "metabolic_syndrome_data" / "raw"

# Type alias for convenient annotations
PathLike = Union[str, Path]


def project_path(path: Optional[PathLike]) -> Optional[Path]:
    """Return an absolute path, mapping relative inputs to the project layout.

    Args:
        path: Absolute or project-relative path. ``None`` is returned unchanged.
    """
    if path is None:
        return None

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate

    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (PROJECT_ROOT / candidate).resolve()
