"""Shared filesystem paths for runtime artifacts."""

from pathlib import Path


# Store traces inside the repository so they are easy to inspect and keep with the project.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACE_DIR = PROJECT_ROOT / "trace"
