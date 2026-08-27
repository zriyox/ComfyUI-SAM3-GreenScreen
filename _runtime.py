"""Runtime dependency resolution for the standalone ComfyUI custom node."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def resolve_ffmpeg_bin(configured: str | None, legacy: Path | None) -> Path:
    """Resolve an executable FFmpeg path without silently ignoring bad config."""

    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
        raise RuntimeError(f"SAM3_FFMPEG_BIN 不可执行: {candidate}")

    discovered = shutil.which("ffmpeg")
    if discovered:
        candidate = Path(discovered)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate

    if legacy is not None:
        candidate = Path(legacy).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate

    raise RuntimeError("FFmpeg 不存在，请安装 ffmpeg 或配置 SAM3_FFMPEG_BIN")
