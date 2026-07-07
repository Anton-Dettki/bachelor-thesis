"""Filesystem paths for federated coordinator artifacts."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

FPM_DIR = Path(__file__).resolve().parent
DEFAULT_GROUPED_OUTPUT_DIR = FPM_DIR / "outputs" / "grouped"
FALLBACK_GROUPED_OUTPUT_DIR = Path(tempfile.gettempdir()) / "fpm" / "outputs" / "grouped"


def resolve_grouped_output_dir(path: Path | str | None = None) -> Path:
    """Return a writable grouped-output directory, falling back if needed."""
    if path is None:
        raw = os.getenv("GROUPED_OUTPUT_DIR", "").strip()
        candidates = [Path(raw)] if raw else []
    else:
        candidates = [Path(path)]

    candidates.extend([DEFAULT_GROUPED_OUTPUT_DIR, FALLBACK_GROUPED_OUTPUT_DIR])

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser()
        if not resolved.is_absolute():
            resolved = (FPM_DIR.parent / resolved).resolve()
        else:
            resolved = resolved.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if _try_prepare_output_dir(resolved):
            return resolved

    raise OSError(
        "Could not create a writable grouped output directory. "
        f"Tried: {', '.join(str(item) for item in seen)}"
    )


def _try_prepare_output_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError:
        return False
    return True
