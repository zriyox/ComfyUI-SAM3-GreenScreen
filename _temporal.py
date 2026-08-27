"""Pure temporal helpers shared by the ComfyUI video-mask nodes."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence

import numpy as np


def detect_shot_starts(
    frame_differences: Sequence[float],
    absolute_threshold: float,
    *,
    frame_samples: np.ndarray | None = None,
    persistence_window: int = 3,
    minimum_persistent_change_fraction: float = 0.35,
    relative_mad_multiplier: float = 16.0,
    minimum_threshold_fraction: float = 0.25,
    min_shot_frames: int = 6,
) -> list[int]:
    """Detect isolated scene changes without treating ordinary motion as cuts."""

    differences = np.asarray(frame_differences, dtype=np.float64)
    if differences.ndim != 1:
        raise ValueError("frame_differences 必须是一维数组")
    if differences.size == 0:
        return []
    if not np.isfinite(differences).all() or np.any(differences < 0):
        raise ValueError("frame_differences 必须是有限非负数")

    threshold = float(absolute_threshold)
    if not np.isfinite(threshold):
        raise ValueError("absolute_threshold 必须是有限数")
    if threshold <= 0:
        return []
    threshold = min(1.0, threshold)

    mad_multiplier = float(relative_mad_multiplier)
    threshold_fraction = float(minimum_threshold_fraction)
    window = int(persistence_window)
    persistent_fraction = float(minimum_persistent_change_fraction)
    spacing = int(min_shot_frames)
    if not np.isfinite(mad_multiplier) or mad_multiplier < 0:
        raise ValueError("relative_mad_multiplier 不能小于零")
    if not np.isfinite(threshold_fraction) or not 0 < threshold_fraction <= 1:
        raise ValueError("minimum_threshold_fraction 必须在零到一之间")
    if window < 1:
        raise ValueError("persistence_window 必须大于零")
    if not np.isfinite(persistent_fraction) or not 0 <= persistent_fraction <= 1:
        raise ValueError("minimum_persistent_change_fraction 必须在零到一之间")
    if spacing < 1:
        raise ValueError("min_shot_frames 必须大于零")

    samples: np.ndarray | None = None
    if frame_samples is not None:
        samples = np.asarray(frame_samples, dtype=np.float32)
        if samples.ndim != 4:
            raise ValueError("frame_samples 必须是 NCHW 四维数组")
        if samples.shape[0] != differences.size + 1:
            raise ValueError("frame_samples 帧数必须比 frame_differences 多一帧")
        if samples.shape[1] < 1 or samples.shape[2] < 2 or samples.shape[3] < 2:
            raise ValueError("frame_samples 必须包含有效图像")
        if not np.isfinite(samples).all():
            raise ValueError("frame_samples 必须是有限数")

    median = float(np.median(differences))
    mad = float(np.median(np.abs(differences - median)))
    robust_sigma = 1.4826 * mad
    adaptive_threshold = max(
        threshold * threshold_fraction,
        median + mad_multiplier * robust_sigma,
    )
    effective_threshold = min(threshold, adaptive_threshold)
    candidates = np.flatnonzero(differences >= effective_threshold)

    if samples is not None and candidates.size:
        persistent_candidates: list[int] = []
        for raw_candidate in candidates:
            candidate = int(raw_candidate)
            boundary = candidate + 1
            pre_start = max(0, boundary - window)
            post_end = min(int(samples.shape[0]), boundary + window)
            pre_frame = np.mean(samples[pre_start:boundary], axis=0)
            persistent_pixel_change = float(np.median([
                np.mean(np.abs(pre_frame - samples[frame_index]))
                for frame_index in range(boundary, post_end)
            ]))
            if persistent_pixel_change >= differences[candidate] * persistent_fraction:
                persistent_candidates.append(candidate)
        candidates = np.asarray(persistent_candidates, dtype=np.int64)

    accepted: list[int] = []
    for candidate in sorted(
        (int(value) for value in candidates),
        key=lambda value: (-float(differences[value]), value),
    ):
        if all(abs(candidate - existing) >= spacing for existing in accepted):
            accepted.append(candidate)
    accepted.sort()

    if samples is not None and len(accepted) >= 4:
        import cv2

        _channels, height, width = samples.shape[1:]
        grid_height = min(20, int(height))
        grid_width = min(12, int(width))

        def transition_fingerprint(candidate: int) -> np.ndarray:
            boundary = candidate + 1
            transition = samples[boundary] - samples[boundary - 1]
            pooled: list[np.ndarray] = []
            for grid_y in range(grid_height):
                top = grid_y * int(height) // grid_height
                bottom = (grid_y + 1) * int(height) // grid_height
                for grid_x in range(grid_width):
                    left = grid_x * int(width) // grid_width
                    right = (grid_x + 1) * int(width) // grid_width
                    pooled.append(
                        np.mean(transition[:, top:bottom, left:right], axis=(1, 2))
                    )
            fingerprint = np.concatenate(pooled).astype(np.float64)
            fingerprint -= float(np.mean(fingerprint))
            norm = float(np.linalg.norm(fingerprint))
            if norm > np.finfo(np.float64).eps:
                fingerprint /= norm
            return fingerprint

        def geometric_residual_ratio(candidate: int) -> float:
            boundary = candidate + 1
            pre_start = max(0, boundary - window)
            post_end = min(int(samples.shape[0]), boundary + window)
            pre_frame = np.mean(samples[pre_start:boundary], axis=0)
            post_frame = np.mean(samples[boundary:post_end], axis=0)
            raw_change = float(np.mean(np.abs(pre_frame - post_frame)))
            if raw_change <= np.finfo(np.float32).eps:
                return 1.0
            if pre_frame.shape[0] == 1:
                pre_hwc = np.transpose(pre_frame, (1, 2, 0))
                post_hwc = np.transpose(post_frame, (1, 2, 0))
                pre_gray = pre_hwc[..., 0]
                post_gray = post_hwc[..., 0]
            elif pre_frame.shape[0] >= 3:
                pre_hwc = np.transpose(pre_frame[:3], (1, 2, 0))
                post_hwc = np.transpose(post_frame[:3], (1, 2, 0))
                pre_gray = cv2.cvtColor(pre_hwc, cv2.COLOR_RGB2GRAY)
                post_gray = cv2.cvtColor(post_hwc, cv2.COLOR_RGB2GRAY)
            else:
                pre_hwc = np.transpose(pre_frame, (1, 2, 0))
                post_hwc = np.transpose(post_frame, (1, 2, 0))
                pre_gray = np.mean(pre_hwc, axis=2)
                post_gray = np.mean(post_hwc, axis=2)
            warp = np.eye(2, 3, dtype=np.float32)
            try:
                _correlation, warp = cv2.findTransformECC(
                    pre_gray,
                    post_gray,
                    warp,
                    cv2.MOTION_AFFINE,
                    (
                        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        80,
                        1e-5,
                    ),
                    None,
                    3,
                )
            except cv2.error:
                return 1.0
            aligned = cv2.warpAffine(
                post_hwc,
                warp,
                (int(width), int(height)),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT,
            )
            return float(np.mean(np.abs(pre_hwc - aligned))) / raw_change

        fingerprints = np.stack([
            transition_fingerprint(candidate)
            for candidate in accepted
        ])
        similarities = fingerprints @ fingerprints.T
        remaining = set(range(len(accepted)))
        repeated_geometric_positions: set[int] = set()
        while remaining:
            seed = remaining.pop()
            component = {seed}
            frontier = [seed]
            while frontier:
                current = frontier.pop()
                neighbors = {
                    position
                    for position in remaining
                    if similarities[current, position] >= 0.9
                }
                remaining.difference_update(neighbors)
                component.update(neighbors)
                frontier.extend(neighbors)
            if len(component) < 4:
                continue
            residuals = [
                geometric_residual_ratio(accepted[position])
                for position in component
            ]
            if float(np.median(residuals)) <= 0.9:
                repeated_geometric_positions.update(component)
        accepted = [
            candidate
            for position, candidate in enumerate(accepted)
            if position not in repeated_geometric_positions
        ]

    return [index + 1 for index in accepted]


def assign_keyframes_to_anchor_segments(
    key_indices: Sequence[int],
    anchor_frames: Sequence[int],
    shot_starts: Sequence[int],
    active_anchors: Sequence[bool] | None = None,
    *,
    reacquire_future_shots: bool = False,
) -> list[list[tuple[int, int]]]:
    """Assign each keyframe to an anchor without crossing shot boundaries.

    Within a shot, frames before its first anchor use that first anchor's semantic
    prompt. Frames at or after an anchor use the latest anchor in the same shot.
    By default, a shot without an anchor remains unassigned so a later shot
    cannot leak backwards into it. When ``reacquire_future_shots`` is enabled,
    shots after the first anchor may reuse the latest prior anchor's semantic
    prompt for full-frame reacquisition; shots before the first anchor remain
    unassigned.
    """

    normalized_anchors = [int(value) for value in anchor_frames]
    if not normalized_anchors:
        return []
    if normalized_anchors != sorted(normalized_anchors):
        raise ValueError("anchor_frames 必须按时间升序排列")
    if len(set(normalized_anchors)) != len(normalized_anchors):
        raise ValueError("anchor_frames 不能重复")

    if active_anchors is None:
        normalized_active = [True] * len(normalized_anchors)
    else:
        normalized_active = [bool(value) for value in active_anchors]
        if len(normalized_active) != len(normalized_anchors):
            raise ValueError("active_anchors 数量必须与 anchor_frames 一致")

    normalized_shot_starts = sorted({0, *(max(0, int(value)) for value in shot_starts)})

    def shot_position(frame_index: int) -> int:
        return bisect_right(normalized_shot_starts, frame_index) - 1

    anchors_by_shot: dict[int, list[int]] = {}
    for anchor_position, frame_index in enumerate(normalized_anchors):
        anchors_by_shot.setdefault(shot_position(frame_index), []).append(anchor_position)

    assignments: list[list[tuple[int, int]]] = [[] for _ in normalized_anchors]
    for key_position, raw_frame_index in enumerate(key_indices):
        frame_index = int(raw_frame_index)
        shot_anchor_positions = anchors_by_shot.get(shot_position(frame_index))
        if not shot_anchor_positions:
            if not reacquire_future_shots:
                continue
            anchor_position = bisect_right(normalized_anchors, frame_index) - 1
            if anchor_position < 0:
                continue
        else:
            shot_anchor_frames = [
                normalized_anchors[anchor_position]
                for anchor_position in shot_anchor_positions
            ]
            relative_position = bisect_right(shot_anchor_frames, frame_index) - 1
            if relative_position < 0:
                anchor_position = shot_anchor_positions[0]
            else:
                anchor_position = shot_anchor_positions[relative_position]

        if normalized_active[anchor_position]:
            assignments[anchor_position].append((key_position, frame_index))

    return assignments


def select_keyframe_mask(
    detected_mask: np.ndarray,
    propagated_mask: np.ndarray,
    *,
    preserve_propagated: bool = False,
) -> np.ndarray:
    """Keep accepted keyframe detections authoritative over optical-flow history.

    Gate-accepted non-empty detections contain the current frame's topology.  Optical
    flow is only a fallback for an empty detection; unioning it into a valid detection
    fills moving background gaps and preserves detached ghost components.
    """

    detected = np.asarray(detected_mask)
    propagated = np.asarray(propagated_mask)
    if detected.shape != propagated.shape:
        raise ValueError("detected_mask 与 propagated_mask 尺寸必须一致")
    if np.any(detected > 0.5):
        selected = np.maximum(detected, propagated) if preserve_propagated else detected
    else:
        selected = propagated
    return np.array(selected, copy=True)


def order_anchor_segment_for_detection(
    assigned_keyframes: Sequence[tuple[int, int]],
    *,
    anchor_frame: int,
) -> list[tuple[int, int]]:
    """Process an anchor first, then walk older frames backwards before future frames."""

    assigned = [(int(position), int(frame)) for position, frame in assigned_keyframes]
    anchor_items = [item for item in assigned if item[1] == int(anchor_frame)]
    if not anchor_items:
        return assigned
    pre_anchor = [item for item in assigned if item[1] < int(anchor_frame)]
    future = [item for item in assigned if item[1] > int(anchor_frame)]
    return [*anchor_items, *reversed(pre_anchor), *future]


def anchor_visibility_allows(
    appearance_distance: float | None,
    threshold: float,
) -> bool:
    """Allow backward anchor propagation only while the target stays visible."""

    limit = float(threshold)
    if limit <= 0:
        return True
    if appearance_distance is None:
        return False
    return float(appearance_distance) <= limit


def select_active_tracking_anchor(
    current_anchor_frame: int | None,
    frame_index: int,
    *,
    mask_nonempty: bool,
    hard_cut: bool,
) -> int | None:
    """Keep grace propagation tied to the latest valid detection in one shot."""

    if hard_cut:
        return int(frame_index) if mask_nonempty else None
    if mask_nonempty:
        return int(frame_index)
    return current_anchor_frame


def temporal_median_preserving_indices(
    masks: np.ndarray,
    shot_starts: Sequence[int],
    radius: int,
    preserve_indices: Sequence[int],
) -> np.ndarray:
    """Apply per-shot temporal median while restoring authoritative frames."""

    source = np.asarray(masks)
    if source.ndim < 1:
        raise ValueError("masks 至少需要一个帧维度")
    frame_count = int(source.shape[0])
    if frame_count == 0:
        return np.array(source, copy=True)

    window_radius = max(1, int(radius))
    normalized_starts = sorted(
        {0, *(int(value) for value in shot_starts if 0 <= int(value) < frame_count)}
    )
    boundaries = [*normalized_starts, frame_count]
    smoothed = np.empty_like(source)
    for start, end in zip(boundaries, boundaries[1:]):
        for frame_index in range(start, end):
            lo = max(start, frame_index - window_radius)
            hi = min(end, frame_index + window_radius + 1)
            smoothed[frame_index] = np.median(source[lo:hi], axis=0)

    for frame_index in {int(value) for value in preserve_indices}:
        if 0 <= frame_index < frame_count:
            smoothed[frame_index] = source[frame_index]
    return smoothed
