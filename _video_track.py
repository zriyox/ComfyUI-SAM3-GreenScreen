"""基于 ComfyUI core SAM3.1 video tracker 的完整视频遮罩分段计划。

纯逻辑模块：只做 anchor 帧解析与追踪分段规划，不依赖 comfy 运行时，可本地单测。
"""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class TrackSegment:
    """一次 core tracker 调用覆盖的帧区间。

    start/stop 为原始帧序（左闭右开语义见 emit_start/emit_stop）。
    reverse=True 表示该段以 anchor 帧为种子向前倒放追踪。
    anchor_index 指向提供 initial mask 的 anchor（None = 纯文本追踪）。
    """

    start: int
    stop: int
    anchor_index: int | None
    reverse: bool
    emit_start: int
    emit_stop: int


def parse_anchor_frames(raw_value: str, frame_count: int) -> list[int]:
    if frame_count <= 0:
        raise ValueError("帧数必须为正")
    try:
        parsed = json.loads(raw_value or "[]")
    except json.JSONDecodeError as exception:
        raise ValueError("anchor_frames_json 不是合法 JSON") from exception
    if not isinstance(parsed, list):
        raise ValueError("anchor_frames_json 必须是整数数组")
    frames: list[int] = []
    for item in parsed:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError("anchor_frames_json 必须是整数数组")
        frames.append(min(max(item, 0), frame_count - 1))
    if len(set(frames)) != len(frames):
        raise ValueError("anchor 帧不能重复")
    return frames


def plan_tracking_segments(
    frame_count: int,
    anchor_frames: list[int],
) -> list[TrackSegment]:
    """把完整视频拆成若干段 core tracker 调用。

    - 无 anchor：整条视频一段正向纯文本追踪。
    - 有 anchor：每个 anchor 作为其后区间的种子正向追踪；
      第一个 anchor 之前的帧用第一个 anchor 倒放回溯。
    """

    if frame_count <= 0:
        raise ValueError("帧数必须为正")
    if not anchor_frames:
        return [
            TrackSegment(
                start=0,
                stop=frame_count,
                anchor_index=None,
                reverse=False,
                emit_start=0,
                emit_stop=frame_count,
            )
        ]

    order = sorted(range(len(anchor_frames)), key=lambda index: anchor_frames[index])
    sorted_frames = [anchor_frames[index] for index in order]
    segments: list[TrackSegment] = []
    first_frame = sorted_frames[0]
    if first_frame > 0:
        segments.append(
            TrackSegment(
                start=0,
                stop=first_frame + 1,
                anchor_index=order[0],
                reverse=True,
                emit_start=0,
                emit_stop=first_frame,
            )
        )
    for position, anchor_index in enumerate(order):
        segment_start = sorted_frames[position]
        segment_stop = (
            sorted_frames[position + 1]
            if position + 1 < len(sorted_frames)
            else frame_count
        )
        segments.append(
            TrackSegment(
                start=segment_start,
                stop=segment_stop,
                anchor_index=anchor_index,
                reverse=False,
                emit_start=segment_start,
                emit_stop=segment_stop,
            )
        )
    return segments
