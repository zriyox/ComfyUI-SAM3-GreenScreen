"""Pure helpers for deriving bounded, grace-limited tracking search regions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math

import numpy as np


class TrackingRoiSource(StrEnum):
    MASK = "MASK"
    PREVIOUS_GRACE = "PREVIOUS_GRACE"
    FULL_FRAME = "FULL_FRAME"


@dataclass(frozen=True)
class TrackingRoi:
    x: int
    y: int
    width: int
    height: int
    source: TrackingRoiSource
    miss_count: int = 0


def derive_tracking_roi(
    mask: np.ndarray,
    previous_roi: TrackingRoi | None,
    frame_width: int,
    frame_height: int,
    margin: float,
    *,
    max_grace_frames: int = 2,
    reset: bool = False,
) -> TrackingRoi:
    """Derive one bounded search ROI from the current mask or short-term history.

    ``margin`` is the fractional total expansion applied to each mask-bbox axis.
    Empty masks may reuse the previous ROI only for ``max_grace_frames`` calls.
    ``reset`` discards that history, which is required at a hard cut.
    """

    width = int(frame_width)
    height = int(frame_height)
    if width <= 0 or height <= 0:
        raise ValueError("画面尺寸必须大于零")

    normalized_margin = float(margin)
    if not math.isfinite(normalized_margin) or not 0 <= normalized_margin <= 4:
        raise ValueError("margin 必须在零到四之间")

    grace_limit = int(max_grace_frames)
    if grace_limit < 0:
        raise ValueError("max_grace_frames 不能小于零")

    normalized_mask = np.asarray(mask)
    if normalized_mask.ndim != 2:
        raise ValueError("mask 必须是二维数组")
    if tuple(normalized_mask.shape) != (height, width):
        raise ValueError("mask 尺寸必须与画面尺寸一致")

    foreground_y, foreground_x = np.nonzero(normalized_mask > 0.5)
    if foreground_x.size > 0:
        left = int(foreground_x.min())
        right = int(foreground_x.max()) + 1
        top = int(foreground_y.min())
        bottom = int(foreground_y.max()) + 1
        return _expand_bbox(
            left,
            top,
            right,
            bottom,
            width,
            height,
            normalized_margin,
        )

    miss_count = (previous_roi.miss_count + 1) if previous_roi is not None else 1
    if not reset and previous_roi is not None and miss_count <= grace_limit:
        return TrackingRoi(
            x=previous_roi.x,
            y=previous_roi.y,
            width=previous_roi.width,
            height=previous_roi.height,
            source=TrackingRoiSource.PREVIOUS_GRACE,
            miss_count=miss_count,
        )

    return TrackingRoi(
        x=0,
        y=0,
        width=width,
        height=height,
        source=TrackingRoiSource.FULL_FRAME,
        miss_count=miss_count,
    )


def derive_tracking_seed_roi(
    mask: np.ndarray,
    initial_roi: TrackingRoi,
    frame_width: int,
    frame_height: int,
    margin: float,
) -> TrackingRoi:
    """Choose a confirmed-mask ROI seed, falling back to the initial ROI."""

    derived = derive_tracking_roi(
        mask,
        previous_roi=None,
        frame_width=frame_width,
        frame_height=frame_height,
        margin=margin,
    )
    if derived.source is TrackingRoiSource.MASK:
        return derived
    return initial_roi


def select_tracking_roi_for_keyframe(
    previous_mask: np.ndarray | None,
    previous_roi: TrackingRoi | None,
    confirmed_seed_roi: TrackingRoi,
    frame_width: int,
    frame_height: int,
    margin: float,
    *,
    max_grace_frames: int = 2,
    hard_cut: bool = False,
    reacquire_on_hard_cut: bool = False,
    hard_boundary: bool = False,
) -> TrackingRoi:
    """Select the search ROI for one keyframe without leaking across hard cuts."""

    if hard_cut:
        if not reacquire_on_hard_cut or hard_boundary:
            return confirmed_seed_roi
        return derive_tracking_roi(
            np.zeros((int(frame_height), int(frame_width)), dtype=np.float32),
            previous_roi,
            frame_width=frame_width,
            frame_height=frame_height,
            margin=margin,
            max_grace_frames=max_grace_frames,
            reset=True,
        )
    if previous_mask is None:
        return previous_roi or confirmed_seed_roi
    return derive_tracking_roi(
        previous_mask,
        previous_roi,
        frame_width=frame_width,
        frame_height=frame_height,
        margin=margin,
        max_grace_frames=max_grace_frames,
    )


def _expand_bbox(
    left: int,
    top: int,
    right: int,
    bottom: int,
    frame_width: int,
    frame_height: int,
    margin: float,
) -> TrackingRoi:
    raw_width = right - left
    raw_height = bottom - top
    expanded_width = min(frame_width, max(1, math.ceil(raw_width * (1 + margin))))
    expanded_height = min(frame_height, max(1, math.ceil(raw_height * (1 + margin))))
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    bounded_x = _bounded_start(center_x, expanded_width, frame_width)
    bounded_y = _bounded_start(center_y, expanded_height, frame_height)
    return TrackingRoi(
        x=bounded_x,
        y=bounded_y,
        width=expanded_width,
        height=expanded_height,
        source=TrackingRoiSource.MASK,
        miss_count=0,
    )


def _bounded_start(center: float, size: int, limit: int) -> int:
    start = math.floor(center - size / 2.0)
    return max(0, min(limit - size, start))
