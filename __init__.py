import io as stdlib_io
import json
import math
import os
import re
import subprocess
import uuid
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import folder_paths
import comfy.model_management
import comfy.utils
from comfy_api.latest import ComfyExtension, io, ui
from comfy_extras.nodes_sam3 import _extract_text_prompts, _refine_mask
from ._runtime import resolve_ffmpeg_bin
from ._identity import (
    masked_color_distance,
    masked_color_signature_distance,
    masked_feature_cosine_similarity,
    select_identity_candidate,
)
from ._video_track import (
    parse_anchor_frames,
    plan_tracking_segments,
)
from ._temporal import (
    anchor_visibility_allows,
    assign_keyframes_to_anchor_segments,
    detect_shot_starts,
    order_anchor_segment_for_detection,
    select_active_tracking_anchor,
    select_keyframe_mask,
    temporal_median_preserving_indices,
)
from .tracking_roi import (
    TrackingRoi,
    TrackingRoiSource,
    derive_tracking_roi,
    derive_tracking_seed_roi,
    select_tracking_roi_for_keyframe,
)


NODE_ROOT = Path(__file__).resolve().parent
LEGACY_FFMPEG_BIN = Path.home() / "ai" / "bin" / "ffmpeg"
PRE_ANCHOR_IDENTITY_MIN_COSINE = 0.55
PRE_ANCHOR_MAX_APPEARANCE_DISTANCE = 0.025


def _parse_hex_color(value: str) -> tuple[float, float, float]:
    text = (value or "#00B140").strip()
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", text):
        text = "#00B140"
    return tuple(int(text[index:index + 2], 16) / 255.0 for index in (1, 3, 5))


def _read_local_setting(name: str) -> str | None:
    config_path = NODE_ROOT / ".env.local"
    if not config_path.exists():
        return None
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == name:
            return value.strip().strip('"').strip("'") or None
    return None


def _resolve_ffmpeg_bin() -> Path:
    configured = os.environ.get("SAM3_FFMPEG_BIN") or _read_local_setting("SAM3_FFMPEG_BIN")
    return resolve_ffmpeg_bin(configured, LEGACY_FFMPEG_BIN)


def _number(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class SAM3BoundingBoxes(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3_BoundingBoxes",
            display_name="SAM3 Bounding Boxes",
            category="image/detection",
            inputs=[
                io.String.Input(
                    "boxes_json",
                    display_name="boxes_json",
                    default="[]",
                    multiline=True,
                ),
            ],
            outputs=[io.BoundingBox.Output("bboxes", display_name="bboxes")],
        )

    @classmethod
    def execute(cls, boxes_json="[]") -> io.NodeOutput:
        import json

        try:
            boxes = json.loads(boxes_json or "[]")
        except json.JSONDecodeError as error:
            raise ValueError(f"框选数据不是合法 JSON: {error}") from error
        if not isinstance(boxes, list):
            raise ValueError("框选数据必须是数组")

        normalized = []
        for index, box in enumerate(boxes):
            if not isinstance(box, dict):
                raise ValueError(f"第 {index + 1} 个框不是对象")
            try:
                x = int(round(float(box["x"])))
                y = int(round(float(box["y"])))
                width = int(round(float(box["width"])))
                height = int(round(float(box["height"])))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"第 {index + 1} 个框缺少 x/y/width/height") from error
            if width <= 0 or height <= 0:
                raise ValueError(f"第 {index + 1} 个框宽高必须大于 0")
            normalized.append({"x": max(0, x), "y": max(0, y), "width": width, "height": height})
        return io.NodeOutput(normalized)


class SAM3ValidateMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3_ValidateMask",
            display_name="Validate SAM3 Mask",
            category="image/detection",
            inputs=[
                io.Mask.Input("mask", display_name="mask"),
                io.Float.Input("min_ratio", display_name="min_ratio", default=0.001, min=0.0, max=1.0, step=0.001),
                io.Float.Input("max_ratio", display_name="max_ratio", default=0.95, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[io.Mask.Output("mask", display_name="mask")],
        )

    @classmethod
    def execute(cls, mask, min_ratio=0.001, max_ratio=0.95) -> io.NodeOutput:
        foreground_ratio = float((mask.detach().float() > 0.5).float().mean().cpu())
        if foreground_ratio < float(min_ratio):
            raise ValueError(
                f"首帧遮罩为空（前景占比 {foreground_ratio:.6f}）。请增加目标正点、缩小框选，或确认框选落在目标上。"
            )
        if foreground_ratio > float(max_ratio):
            raise ValueError(
                f"首帧遮罩几乎覆盖全画面（前景占比 {foreground_ratio:.4f}）。请缩小框选或增加背景负点。"
            )
        return io.NodeOutput(mask)


class SAM3MultiPromptDetectCached(io.ComfyNode):
    """Detect multiple text-prompted parts while sharing one vision trunk pass per frame batch."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3_MultiPromptDetectCached",
            display_name="SAM3 Multi-Prompt Detect (Cached Vision)",
            category="image/detection",
            inputs=[
                io.Model.Input("model", display_name="model"),
                io.Image.Input("image", display_name="image"),
                io.Conditioning.Input("conditioning", display_name="conditioning"),
                io.Float.Input(
                    "threshold", display_name="threshold",
                    default=0.5, min=0.0, max=1.0, step=0.01,
                ),
                io.Int.Input(
                    "refine_iterations", display_name="refine_iterations",
                    default=1, min=0, max=5,
                ),
                io.Int.Input(
                    "frame_batch_size", display_name="frame_batch_size",
                    default=1, min=1, max=8,
                ),
            ],
            outputs=[
                io.Mask.Output("masks", display_name="masks"),
                io.BoundingBox.Output("bboxes", display_name="bboxes"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        image,
        conditioning,
        threshold=0.5,
        refine_iterations=1,
        frame_batch_size=1,
    ) -> io.NodeOutput:
        frame_count, height, width, _channels = image.shape
        image_in = comfy.utils.common_upscale(
            image[..., :3].movedim(-1, 1),
            1008,
            1008,
            "bilinear",
            crop="disabled",
        )

        comfy.model_management.load_model_gpu(model)
        device = comfy.model_management.get_torch_device()
        dtype = model.model.get_dtype()
        sam3_model = model.model.diffusion_model
        detector = sam3_model.detector
        vision_backbone = detector.backbone["vision_backbone"]
        text_resizer = detector.backbone["language_backbone"]["resizer"]

        prompts = _extract_text_prompts(conditioning, device, dtype)
        if not prompts:
            raise ValueError("至少需要一个 SAM3 文本提示词")
        resized_prompts = [
            (
                text_resizer(text_embeddings),
                text_mask.bool() if text_mask is not None else None,
                max_detections,
            )
            for text_embeddings, text_mask, max_detections in prompts
        ]

        batch_size = max(1, min(int(frame_batch_size), frame_count))
        all_bbox_dicts = []
        all_masks = []
        pbar = comfy.utils.ProgressBar(frame_count)
        scale = torch.tensor(
            [width, height, width, height],
            device=device,
            dtype=dtype,
        )

        for chunk_start in range(0, frame_count, batch_size):
            chunk_end = min(frame_count, chunk_start + batch_size)
            frames = image_in[chunk_start:chunk_end].to(device=device, dtype=dtype)
            trunk_out = vision_backbone.trunk(frames)
            prompt_results = []
            for text_embeddings, text_mask, max_detections in resized_prompts:
                result = detector.forward_from_trunk(trunk_out, text_embeddings, text_mask)
                prompt_results.append((
                    result["boxes"] * scale,
                    result["scores"].sigmoid(),
                    F.interpolate(
                        result["masks"],
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False,
                    ),
                    max_detections,
                ))

            for local_index, frame_index in enumerate(range(chunk_start, chunk_end)):
                frame_bbox_dicts = []
                frame_masks = []
                for boxes, probabilities, masks, max_detections in prompt_results:
                    frame_boxes = boxes[local_index]
                    frame_probabilities = probabilities[local_index]
                    frame_masks_for_prompt = masks[local_index]
                    keep = frame_probabilities > threshold
                    kept_boxes = frame_boxes[keep]
                    kept_probabilities = frame_probabilities[keep]
                    kept_masks = frame_masks_for_prompt[keep]
                    order = kept_probabilities.argsort(descending=True)[:max_detections]
                    kept_boxes = kept_boxes[order]
                    kept_probabilities = kept_probabilities[order]
                    kept_masks = kept_masks[order]

                    for box, score in zip(kept_boxes, kept_probabilities):
                        cpu_box = box.detach().cpu()
                        frame_bbox_dicts.append({
                            "x": float(cpu_box[0]),
                            "y": float(cpu_box[1]),
                            "width": float(cpu_box[2] - cpu_box[0]),
                            "height": float(cpu_box[3] - cpu_box[1]),
                            "score": float(score.detach().cpu()),
                        })
                    for coarse_mask, box in zip(kept_masks, kept_boxes):
                        frame_masks.append(_refine_mask(
                            sam3_model,
                            image[frame_index],
                            coarse_mask,
                            box,
                            height,
                            width,
                            device,
                            dtype,
                            int(refine_iterations),
                        ))

                all_bbox_dicts.append(frame_bbox_dicts)
                if frame_masks:
                    combined = torch.cat(frame_masks, dim=0)
                    all_masks.append((combined > 0).any(dim=0).float())
                else:
                    all_masks.append(torch.zeros(height, width, device=device, dtype=torch.float32))
                pbar.update(1)

            del trunk_out, prompt_results

        intermediate_device = comfy.model_management.intermediate_device()
        mask_out = torch.stack([mask.to(intermediate_device) for mask in all_masks])
        return io.NodeOutput(mask_out, all_bbox_dicts)


class SAM3CleanMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3_CleanMask",
            display_name="Clean SAM3 Mask",
            category="image/detection",
            inputs=[
                io.Mask.Input("mask", display_name="mask"),
                io.Int.Input("min_component_area", display_name="min_component_area", default=128, min=0, max=100000),
                io.Int.Input("max_hole_area", display_name="max_hole_area", default=1500, min=0, max=100000),
                io.Int.Input("close_radius", display_name="close_radius", default=2, min=0, max=12),
                io.Int.Input("smooth_radius", display_name="smooth_radius", default=2, min=0, max=8),
                io.Int.Input("expand_pixels", display_name="expand_pixels", default=1, min=0, max=12),
                io.Float.Input("feather_radius", display_name="feather_radius", default=0.8, min=0.0, max=8.0, step=0.1),
            ],
            outputs=[io.Mask.Output("clean_mask", display_name="clean_mask")],
        )

    @classmethod
    def execute(cls, mask, min_component_area=128, max_hole_area=1500,
                close_radius=2, smooth_radius=2, expand_pixels=1, feather_radius=0.8) -> io.NodeOutput:
        from scipy import ndimage

        source = mask.detach().float().cpu().numpy()
        cleaned_frames = []
        for frame in source:
            binary = frame > 0.5

            if close_radius > 0:
                radius = int(close_radius)
                yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
                structure = (xx * xx + yy * yy) <= radius * radius
                binary = ndimage.binary_closing(binary, structure=structure)

            # Remove one-pixel spurs and staircase noise without filling the
            # larger negative-point exclusion areas (face/hand remain intact).
            if smooth_radius > 0:
                radius = int(smooth_radius)
                yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
                structure = (xx * xx + yy * yy) <= radius * radius
                binary = ndimage.binary_opening(binary, structure=structure)
                binary = ndimage.median_filter(binary.astype(np.uint8), size=3) > 0

            if min_component_area > 0:
                labels, count = ndimage.label(binary)
                if count > 0:
                    sizes = np.bincount(labels.ravel())
                    keep = sizes >= int(min_component_area)
                    keep[0] = False
                    binary = keep[labels]

            if max_hole_area > 0:
                background = ~binary
                labels, count = ndimage.label(background)
                if count > 0:
                    border_labels = np.unique(np.concatenate((
                        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]
                    )))
                    sizes = np.bincount(labels.ravel())
                    fill = sizes <= int(max_hole_area)
                    fill[0] = False
                    fill[border_labels] = False
                    binary = binary | fill[labels]

            if expand_pixels > 0:
                binary = ndimage.binary_dilation(binary, iterations=int(expand_pixels))

            result = binary.astype(np.float32)
            if feather_radius > 0:
                result = ndimage.gaussian_filter(result, sigma=float(feather_radius))
                result = np.clip(result, 0.0, 1.0)
            cleaned_frames.append(result)

        clean_mask = torch.from_numpy(np.stack(cleaned_frames)).to(
            device=mask.device, dtype=mask.dtype
        )
        return io.NodeOutput(clean_mask)


class SAM3ClipMaskToBox(io.ComfyNode):
    """Hard-limit a SAM3 mask to the user-selected image box."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3_ClipMaskToBox",
            display_name="Clip SAM3 Mask to Box",
            category="image/detection",
            inputs=[
                io.Mask.Input("mask", display_name="mask"),
                io.String.Input(
                    "boxes_json",
                    display_name="boxes_json",
                    default="[]",
                    multiline=True,
                ),
            ],
            outputs=[io.Mask.Output("clipped_mask", display_name="clipped_mask")],
        )

    @classmethod
    def execute(cls, mask, boxes_json="[]") -> io.NodeOutput:
        import json

        try:
            boxes = json.loads(boxes_json or "[]")
        except json.JSONDecodeError as error:
            raise ValueError(f"裁剪框数据不是合法 JSON: {error}") from error
        if not isinstance(boxes, list) or not boxes:
            raise ValueError("裁剪框不能为空")

        height, width = mask.shape[-2:]
        frame_count = mask.shape[0]
        clipped = torch.zeros_like(mask)
        for frame_index in range(frame_count):
            box = boxes[0] if len(boxes) == 1 else boxes[min(frame_index, len(boxes) - 1)]
            if not isinstance(box, dict):
                raise ValueError("裁剪框必须是对象")
            try:
                left = max(0, min(width, int(round(float(box["x"])))))
                top = max(0, min(height, int(round(float(box["y"])))))
                right = max(left, min(width, int(round(float(box["x"]) + float(box["width"])))))
                bottom = max(top, min(height, int(round(float(box["y"]) + float(box["height"])))) )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("裁剪框缺少 x/y/width/height") from error
            clipped[frame_index, top:bottom, left:right] = mask[frame_index, top:bottom, left:right]
        return io.NodeOutput(clipped)


class SAM3MaskBatch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        inputs = [io.Mask.Input("mask_1", display_name="mask_1")]
        inputs.extend(
            io.Mask.Input(f"mask_{index}", display_name=f"mask_{index}", optional=True)
            for index in range(2, 9)
        )
        return io.Schema(
            node_id="SAM3_MaskBatch",
            display_name="Batch SAM3 Masks",
            category="image/detection",
            inputs=inputs,
            outputs=[io.Mask.Output("masks", display_name="masks")],
        )

    @classmethod
    def execute(cls, mask_1, mask_2=None, mask_3=None, mask_4=None, mask_5=None,
                mask_6=None, mask_7=None, mask_8=None) -> io.NodeOutput:
        masks = [mask_1, mask_2, mask_3, mask_4, mask_5, mask_6, mask_7, mask_8]
        normalized = []
        target_size = tuple(mask_1.shape[-2:])
        for mask in masks:
            if mask is None:
                continue
            current = mask.detach().float()
            if tuple(current.shape[-2:]) != target_size:
                current = F.interpolate(
                    current.unsqueeze(1), size=target_size, mode="bilinear", align_corners=False
                )[:, 0]
            if current.shape[0] != 1:
                current = current.max(dim=0, keepdim=True).values
            normalized.append(current)
        if not normalized:
            raise ValueError("至少需要一个区域遮罩")
        return io.NodeOutput(torch.cat(normalized, dim=0))


def _parse_roles(roles_json: str, object_count: int) -> list[str]:
    import json

    try:
        roles = json.loads(roles_json or "[]")
    except json.JSONDecodeError as error:
        raise ValueError(f"区域角色不是合法 JSON: {error}") from error
    if not isinstance(roles, list) or len(roles) != object_count:
        raise ValueError(f"区域角色数量 {len(roles) if isinstance(roles, list) else 0} 与遮罩数量 {object_count} 不一致")
    normalized = [str(role).upper() for role in roles]
    if any(role not in {"REPLACE", "KEEP"} for role in normalized):
        raise ValueError("区域角色只能是 REPLACE 或 KEEP")
    return normalized


def _compose_edit_mask(masks: torch.Tensor, roles: list[str], replace_background: bool) -> torch.Tensor:
    binary = masks.detach().float().clamp(0.0, 1.0)
    replace_indices = [index for index, role in enumerate(roles) if role == "REPLACE"]
    keep_indices = [index for index, role in enumerate(roles) if role == "KEEP"]

    replace_union = torch.zeros_like(binary[0])
    for index in replace_indices:
        replace_union = torch.maximum(replace_union, binary[index])

    if replace_background:
        if not keep_indices:
            raise ValueError("开启替换背景时，至少添加一个“保留区域”")
        keep_union = torch.zeros_like(binary[0])
        for index in keep_indices:
            keep_union = torch.maximum(keep_union, binary[index])
        edit_mask = 1.0 - keep_union
        edit_mask = torch.maximum(edit_mask, replace_union)
    else:
        if not replace_indices:
            raise ValueError("未开启替换背景时，至少添加一个“变绿区域”")
        edit_mask = replace_union
    return edit_mask.clamp(0.0, 1.0)


class SAM3ComposeEditMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3_ComposeEditMask",
            display_name="Compose SAM3 Edit Mask",
            category="image/detection",
            inputs=[
                io.Mask.Input("masks", display_name="masks"),
                io.String.Input("roles_json", display_name="roles_json", default='["REPLACE"]'),
                io.Boolean.Input("replace_background", display_name="replace_background", default=False),
            ],
            outputs=[io.Mask.Output("edit_mask", display_name="edit_mask")],
        )

    @classmethod
    def execute(cls, masks, roles_json='["REPLACE"]', replace_background=False) -> io.NodeOutput:
        roles = _parse_roles(roles_json, masks.shape[0])
        return io.NodeOutput(_compose_edit_mask(masks, roles, bool(replace_background)).unsqueeze(0))


class SAM3TrackEditMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3_TrackEditMask",
            display_name="SAM3 Track to Edit Mask",
            category="image/detection",
            inputs=[
                io.Custom("SAM3_TRACK_DATA").Input("track_data", display_name="track_data"),
                io.String.Input("roles_json", display_name="roles_json", default='["REPLACE"]'),
                io.Boolean.Input("replace_background", display_name="replace_background", default=False),
            ],
            outputs=[io.Mask.Output("edit_masks", display_name="edit_masks")],
        )

    @classmethod
    def execute(cls, track_data, roles_json='["REPLACE"]', replace_background=False) -> io.NodeOutput:
        from comfy.ldm.sam3.tracker import unpack_masks

        packed = track_data.get("packed_masks")
        if packed is None:
            raise ValueError("视频跟踪没有输出任何区域遮罩")
        masks = unpack_masks(packed).float()
        frame_count, object_count = masks.shape[:2]
        import json
        try:
            requested_roles = json.loads(roles_json or "[]")
        except json.JSONDecodeError as error:
            raise ValueError(f"区域角色不是合法 JSON: {error}") from error
        # A single semantic target can temporarily own multiple tracker slots:
        # one stale propagated track plus one freshly redetected track. Treat
        # those slots as the same edit role and union them. Multi-region flows
        # still require an exact role count to avoid mixing distinct targets.
        if isinstance(requested_roles, list) and len(requested_roles) == 1 and object_count > 1:
            roles = _parse_roles(json.dumps(requested_roles * object_count), object_count)
        else:
            roles = _parse_roles(roles_json, object_count)
        edit_frames = [
            _compose_edit_mask(masks[frame_index], roles, bool(replace_background))
            for frame_index in range(frame_count)
        ]
        edit_masks = torch.stack(edit_frames)
        height, width = track_data["orig_size"]
        edit_masks = F.interpolate(
            edit_masks.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False
        )[:, 0]
        return io.NodeOutput(edit_masks)


def _keyframe_indices(frame_count: int, interval: int) -> list[int]:
    if frame_count <= 0:
        return []
    step = max(1, int(interval))
    indices = list(range(0, frame_count, step))
    if indices[-1] != frame_count - 1:
        indices.append(frame_count - 1)
    return indices


def _scene_shot_starts(images, threshold: float) -> list[int]:
    """Return frame indices that begin a new shot (mean-abs-diff >= threshold)."""
    frame_count = int(images.shape[0])
    if threshold <= 0.0 or frame_count <= 1:
        return []
    _batch, height, width, _channels = images.shape
    longest_side = 160
    scale = min(1.0, longest_side / max(int(height), int(width)))
    sample_height = max(16, int(round(int(height) * scale)))
    sample_width = max(16, int(round(int(width) * scale)))
    samples = F.interpolate(
        images[..., :3].detach().float().permute(0, 3, 1, 2),
        size=(sample_height, sample_width),
        mode="area",
    ).cpu().numpy()
    frame_differences = np.mean(
        np.abs(samples[1:] - samples[:-1]),
        axis=(1, 2, 3),
    )
    return detect_shot_starts(
        frame_differences,
        absolute_threshold=threshold,
        frame_samples=samples,
    )


def _scene_keyframe_indices(images, interval: int, scene_threshold: float) -> list[int]:
    """Select periodic keyframes while forcing every hard-cut boundary into the set."""
    frame_count = int(images.shape[0])
    if frame_count <= 0:
        return []

    threshold = max(0.0, min(1.0, float(scene_threshold)))
    shot_starts = _scene_shot_starts(images, threshold)
    boundaries = [0, *shot_starts, frame_count]

    step = max(1, int(interval))
    indices: set[int] = set()
    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        indices.update(range(start, end, step))
        indices.add(end - 1)
    return sorted(indices)

def _parse_forced_frame_indices(raw_value, frame_count: int) -> list[int]:
    try:
        values = json.loads(raw_value or "[]")
    except json.JSONDecodeError as error:
        raise ValueError(f"强制关键帧不是合法 JSON: {error}") from error
    if not isinstance(values, list):
        raise ValueError("强制关键帧必须是数组")
    normalized = []
    for value in values:
        try:
            frame_index = int(round(float(value)))
        except (TypeError, ValueError) as error:
            raise ValueError("强制关键帧必须是数字") from error
        if frame_count > 0:
            frame_index = max(0, min(frame_count - 1, frame_index))
        normalized.append(frame_index)
    return sorted(set(normalized))


def _anchored_keyframe_indices(images, interval, scene_threshold, forced_frames_json="[]") -> list[int]:
    indices = set(_scene_keyframe_indices(images, int(interval), float(scene_threshold)))
    indices.update(_parse_forced_frame_indices(forced_frames_json, int(images.shape[0])))
    return sorted(indices)


def _tracking_roi_from_box(box: dict, width: int, height: int) -> TrackingRoi:
    left, top, right, bottom = _box_edges(box, width, height)
    return TrackingRoi(
        x=left,
        y=top,
        width=right - left,
        height=bottom - top,
        source=TrackingRoiSource.FULL_FRAME,
    )


def _normalize_confirmed_mask(mask, height: int, width: int, device, dtype):
    normalized = mask.detach().float()
    if normalized.ndim == 2:
        normalized = normalized.unsqueeze(0)
    if normalized.ndim != 3 or normalized.shape[0] < 1:
        raise ValueError("确认遮罩必须是 [帧数, 高, 宽] 格式")
    if normalized.shape[0] > 1:
        normalized = normalized.max(dim=0, keepdim=True).values
    if tuple(normalized.shape[-2:]) != (height, width):
        normalized = F.interpolate(
            normalized.unsqueeze(1),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
    return normalized[0].to(device=device, dtype=dtype).clamp(0.0, 1.0)


def _clip_mask_to_box(mask, box: dict, width: int, height: int):
    left, top, right, bottom = _box_edges(box, width, height)
    clipped = torch.zeros_like(mask)
    clipped[top:bottom, left:right] = mask[top:bottom, left:right]
    return clipped


def _mask_bbox_dict(mask) -> dict | None:
    ys, xs = np.nonzero(mask.detach().float().cpu().numpy() > 0.5)
    if not len(xs):
        return None
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    return {
        "x": float(left),
        "y": float(top),
        "width": float(right - left),
        "height": float(bottom - top),
        "score": 1.0,
    }


def _execute_roi_keyframe_detection(
    *,
    model,
    sam3_model,
    vision_backbone,
    images,
    key_indices: list[int],
    assignments: list[list[tuple[int, int]]],
    anchor_frames: list[int],
    active_anchors: list[bool],
    prepared_prompts,
    confirmed_masks,
    initial_boxes: list[dict],
    threshold: float,
    refine_iterations: int,
    device,
    dtype,
    height: int,
    width: int,
    roi_margin: float,
    roi_max_grace_frames: int,
    roi_constraint_mode: str,
    scene_threshold: float,
    reacquire_on_hard_cut: bool,
) -> tuple[torch.Tensor, list[list[dict]]]:
    all_bbox_dicts: list[list[dict]] = [[] for _ in key_indices]
    all_masks = [torch.zeros(height, width, device=device, dtype=torch.float32) for _ in key_indices]
    assigned_key_positions = {
        key_position
        for anchor_assignments in assignments
        for key_position, _frame_index in anchor_assignments
    }
    pbar = comfy.utils.ProgressBar(len(key_indices))
    for key_position in range(len(key_indices)):
        if key_position not in assigned_key_positions:
            pbar.update(1)

    shot_starts = set(_scene_shot_starts(images, float(scene_threshold)))
    hard_boundary = roi_constraint_mode == "HARD_BOUNDARY"
    for anchor_position, assigned in enumerate(assignments):
        if not active_anchors[anchor_position]:
            continue
        assigned = order_anchor_segment_for_detection(
            assigned,
            anchor_frame=anchor_frames[anchor_position],
        )
        previous_mask = None
        initial_roi = _tracking_roi_from_box(initial_boxes[anchor_position], width, height)
        confirmed_seed_roi = initial_roi
        confirmed_seed_mask = None
        if (
            anchor_position < len(confirmed_masks)
            and confirmed_masks[anchor_position] is not None
        ):
            confirmed_seed_mask = _normalize_confirmed_mask(
                confirmed_masks[anchor_position],
                height,
                width,
                device,
                dtype,
            )
            if hard_boundary:
                confirmed_seed_mask = _clip_mask_to_box(
                    confirmed_seed_mask,
                    initial_boxes[anchor_position],
                    width,
                    height,
                )
            confirmed_seed_roi = derive_tracking_seed_roi(
                confirmed_seed_mask.detach().cpu().numpy(),
                initial_roi,
                width,
                height,
                roi_margin,
            )
        reference_features = None
        reference_mask_feature = None
        has_pre_anchor_keyframes = any(
            frame_index < anchor_frames[anchor_position]
            for _key_position, frame_index in assigned
        )
        if (
            has_pre_anchor_keyframes
            and confirmed_seed_mask is not None
            and not hard_boundary
        ):
            reference_image = images[
                anchor_frames[anchor_position]:anchor_frames[anchor_position] + 1,
                :, :, :3,
            ]
            reference_input = F.interpolate(
                reference_image.permute(0, 3, 1, 2),
                size=(1008, 1008),
                mode="bilinear",
                align_corners=False,
            ).to(device=device, dtype=dtype)
            reference_features = vision_backbone.trunk(reference_input)
            reference_mask_feature = F.interpolate(
                confirmed_seed_mask[None, None],
                size=reference_features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        previous_roi = confirmed_seed_roi
        pre_anchor_tracking_lost = False
        future_tracking_started = False
        for key_position, frame_index in assigned:
            is_pre_anchor = (
                reference_features is not None
                and frame_index < anchor_frames[anchor_position]
            )
            if frame_index > anchor_frames[anchor_position] and not future_tracking_started:
                future_tracking_started = True
                previous_mask = confirmed_seed_mask
                previous_roi = confirmed_seed_roi
            if is_pre_anchor and pre_anchor_tracking_lost:
                all_masks[key_position] = torch.zeros(
                    height,
                    width,
                    device=device,
                    dtype=torch.float32,
                )
                all_bbox_dicts[key_position] = []
                pbar.update(1)
                continue
            is_hard_cut = frame_index in shot_starts
            if is_hard_cut:
                previous_mask = None

            confirmed_mask = (
                confirmed_masks[anchor_position]
                if anchor_position < len(confirmed_masks)
                and frame_index == anchor_frames[anchor_position]
                else None
            )
            if confirmed_mask is not None:
                current_mask = _normalize_confirmed_mask(
                    confirmed_mask,
                    height,
                    width,
                    device,
                    dtype,
                )
                if hard_boundary:
                    current_mask = _clip_mask_to_box(
                        current_mask,
                        initial_boxes[anchor_position],
                        width,
                        height,
                    )
                all_masks[key_position] = current_mask.float()
                bbox = _mask_bbox_dict(current_mask)
                all_bbox_dicts[key_position] = [bbox] if bbox is not None else []
                previous_mask = current_mask
                previous_roi = derive_tracking_roi(
                    current_mask.detach().cpu().numpy(),
                    initial_roi,
                    width,
                    height,
                    roi_margin,
                    max_grace_frames=roi_max_grace_frames,
                )
                pbar.update(1)
                continue

            roi = select_tracking_roi_for_keyframe(
                previous_mask=(
                    previous_mask.detach().cpu().numpy()
                    if previous_mask is not None
                    else None
                ),
                previous_roi=previous_roi,
                confirmed_seed_roi=confirmed_seed_roi,
                frame_width=width,
                frame_height=height,
                margin=roi_margin,
                max_grace_frames=roi_max_grace_frames,
                hard_cut=is_hard_cut,
                reacquire_on_hard_cut=bool(reacquire_on_hard_cut),
                hard_boundary=hard_boundary,
            )
            left, top, right, bottom = _box_edges(
                {
                    "x": roi.x,
                    "y": roi.y,
                    "width": roi.width,
                    "height": roi.height,
                },
                width,
                height,
            )
            crop_height = bottom - top
            crop_width = right - left
            crop = images[frame_index:frame_index + 1, top:bottom, left:right, :3]
            crop_input = F.interpolate(
                crop.permute(0, 3, 1, 2),
                size=(1008, 1008),
                mode="bilinear",
                align_corners=False,
            ).to(device=device, dtype=dtype)
            trunk_out = vision_backbone.trunk(crop_input)
            prompt_results = []
            crop_scale = torch.tensor(
                [crop_width, crop_height, crop_width, crop_height],
                device=device,
                dtype=dtype,
            )
            for text_embeddings, text_mask, max_detections in prepared_prompts[anchor_position]:
                result = sam3_model.detector.forward_from_trunk(
                    trunk_out,
                    text_embeddings,
                    text_mask,
                )
                prompt_results.append((
                    result["boxes"] * crop_scale,
                    result["scores"].sigmoid(),
                    F.interpolate(
                        result["masks"],
                        size=(crop_height, crop_width),
                        mode="bilinear",
                        align_corners=False,
                    ),
                    max_detections,
                ))

            candidate_records = []
            candidate_index = 0
            for boxes, probabilities, masks, max_detections in prompt_results:
                frame_probabilities = probabilities[0]
                frame_boxes = boxes[0]
                frame_masks_for_prompt = masks[0]
                keep = frame_probabilities > float(threshold)
                kept_boxes = frame_boxes[keep]
                kept_probabilities = frame_probabilities[keep]
                kept_masks = frame_masks_for_prompt[keep]
                order = kept_probabilities.argsort(descending=True)[:max_detections]
                kept_boxes = kept_boxes[order]
                kept_probabilities = kept_probabilities[order]
                kept_masks = kept_masks[order]
                for box, score, crop_mask in zip(
                    kept_boxes,
                    kept_probabilities,
                    kept_masks,
                ):
                    crop_left = max(0.0, min(float(crop_width), float(box[0])))
                    crop_top = max(0.0, min(float(crop_height), float(box[1])))
                    crop_right = max(crop_left, min(float(crop_width), float(box[2])))
                    crop_bottom = max(crop_top, min(float(crop_height), float(box[3])))
                    full_box = torch.tensor(
                        [
                            left + crop_left,
                            top + crop_top,
                            left + crop_right,
                            top + crop_bottom,
                        ],
                        device=device,
                        dtype=dtype,
                    )
                    full_mask = torch.zeros(
                        height,
                        width,
                        device=device,
                        dtype=torch.float32,
                    )
                    full_mask[top:bottom, left:right] = crop_mask
                    if hard_boundary:
                        full_mask = _clip_mask_to_box(
                            full_mask,
                            initial_boxes[anchor_position],
                            width,
                            height,
                        )
                    candidate_records.append({
                        "index": candidate_index,
                        "score": float(score.detach().cpu()),
                        "full_box": full_box,
                        "full_mask": full_mask,
                    })
                    candidate_index += 1

            refined_records = []
            for candidate in candidate_records:
                full_box = candidate["full_box"]
                full_mask = candidate["full_mask"]
                refined = _refine_mask(
                    sam3_model,
                    images[frame_index],
                    full_mask,
                    full_box,
                    height,
                    width,
                    device,
                    dtype,
                    int(refine_iterations),
                )
                if refined.ndim == 2:
                    refined = refined.unsqueeze(0)
                if hard_boundary:
                    refined[0] = _clip_mask_to_box(
                        refined[0],
                        initial_boxes[anchor_position],
                        width,
                        height,
                    )
                record = {**candidate, "refined": refined}
                if is_pre_anchor:
                    refined_crop = refined[0, top:bottom, left:right]
                    candidate_mask_feature = F.interpolate(
                        refined_crop[None, None],
                        size=trunk_out.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0]
                    similarity = masked_feature_cosine_similarity(
                        reference_features,
                        reference_mask_feature,
                        trunk_out,
                        candidate_mask_feature,
                    )
                    appearance_distance = masked_color_signature_distance(
                        images[anchor_frames[anchor_position]],
                        confirmed_seed_mask,
                        images[frame_index],
                        (refined[0] > 0).float(),
                    )
                    record.update({
                        "similarity": (
                            -1.0 if similarity is None else float(similarity)
                        ),
                        "appearance_distance": (
                            float("inf")
                            if appearance_distance is None
                            else float(appearance_distance)
                        ),
                    })
                refined_records.append(record)

            if is_pre_anchor:
                selected = select_identity_candidate(
                    refined_records,
                    minimum_similarity=PRE_ANCHOR_IDENTITY_MIN_COSINE,
                    maximum_appearance_distance=PRE_ANCHOR_MAX_APPEARANCE_DISTANCE,
                )
                if selected is None:
                    pre_anchor_tracking_lost = True
                    refined_records = []
                else:
                    refined_records = [selected]

            frame_bbox_dicts = []
            frame_masks = []
            for candidate in refined_records:
                full_box = candidate["full_box"]
                frame_masks.append(candidate["refined"])
                frame_bbox_dicts.append({
                    "x": float(full_box[0].detach().cpu()),
                    "y": float(full_box[1].detach().cpu()),
                    "width": float((full_box[2] - full_box[0]).detach().cpu()),
                    "height": float((full_box[3] - full_box[1]).detach().cpu()),
                    "score": float(candidate["score"]),
                })
            if frame_masks:
                current_mask = torch.cat(frame_masks, dim=0).max(dim=0).values.float()
            else:
                current_mask = torch.zeros(height, width, device=device, dtype=torch.float32)
            all_masks[key_position] = current_mask
            all_bbox_dicts[key_position] = frame_bbox_dicts
            previous_mask = current_mask
            previous_roi = roi
            pbar.update(1)
            del trunk_out, prompt_results

    intermediate_device = comfy.model_management.intermediate_device()
    mask_out = torch.stack([mask.to(intermediate_device) for mask in all_masks])
    return mask_out, all_bbox_dicts


class SAM3AnchoredKeyframeDetectCached(io.ComfyNode):
    """Run one SAM3 text prompt per anchor segment while encoding each keyframe once."""

    @classmethod
    def define_schema(cls):
        inputs = [
            io.Model.Input("model", display_name="model"),
            io.Image.Input("images", display_name="images"),
            io.Conditioning.Input("conditioning_1", display_name="conditioning_1"),
            io.String.Input(
                "anchor_frames_json",
                display_name="anchor_frames_json",
                default="[0]",
                multiline=True,
            ),
            io.String.Input(
                "active_anchors_json",
                display_name="active_anchors_json",
                default="[]",
                multiline=True,
                optional=True,
                tooltip="每个锚点是否启用检测；false 区间输出空 Mask",
            ),
            io.Int.Input("interval", display_name="interval", default=6, min=1, max=120),
            io.Float.Input(
                "scene_threshold", display_name="scene_threshold",
                default=0.12, min=0.0, max=1.0, step=0.01,
            ),
            io.Boolean.Input(
                "reacquire_on_hard_cut",
                display_name="reacquire_on_hard_cut",
                default=True,
            ),
            io.Float.Input(
                "threshold", display_name="threshold",
                default=0.25, min=0.0, max=1.0, step=0.01,
            ),
            io.Int.Input(
                "refine_iterations", display_name="refine_iterations",
                default=1, min=0, max=5,
            ),
            io.Int.Input(
                "frame_batch_size", display_name="frame_batch_size",
                default=1, min=1, max=8,
            ),
            io.Boolean.Input(
                "roi_enabled",
                display_name="roi_enabled",
                default=False,
            ),
            io.Float.Input(
                "roi_margin",
                display_name="roi_margin",
                default=0.6,
                min=0.0,
                max=4.0,
                step=0.1,
            ),
            io.Int.Input(
                "roi_max_grace_frames",
                display_name="roi_max_grace_frames",
                default=2,
                min=0,
                max=16,
            ),
            io.String.Input(
                "roi_constraint_mode",
                display_name="roi_constraint_mode",
                default="SEARCH_ROI",
            ),
            io.String.Input(
                "initial_boxes_json",
                display_name="initial_boxes_json",
                default="[]",
                multiline=True,
            ),
        ]
        inputs.extend(
            io.Conditioning.Input(
                f"conditioning_{index}",
                display_name=f"conditioning_{index}",
                optional=True,
            )
            for index in range(2, 9)
        )
        inputs.extend(
            io.Mask.Input(
                f"confirmed_mask_{index}",
                display_name=f"confirmed_mask_{index}",
                optional=True,
            )
            for index in range(1, 9)
        )
        return io.Schema(
            node_id="SAM3_AnchoredKeyframeDetectCached",
            display_name="SAM3 Anchored Keyframe Detect (Cached Vision)",
            category="image/detection",
            inputs=inputs,
            outputs=[
                io.Mask.Output("masks", display_name="masks"),
                io.BoundingBox.Output("bboxes", display_name="bboxes"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        images,
        conditioning_1,
        anchor_frames_json="[0]",
        active_anchors_json="[]",
        interval=6,
        scene_threshold=0.12,
        reacquire_on_hard_cut=True,
        threshold=0.25,
        refine_iterations=1,
        frame_batch_size=1,
        roi_enabled=False,
        roi_margin=0.6,
        roi_max_grace_frames=2,
        roi_constraint_mode="SEARCH_ROI",
        initial_boxes_json="[]",
        conditioning_2=None,
        conditioning_3=None,
        conditioning_4=None,
        conditioning_5=None,
        conditioning_6=None,
        conditioning_7=None,
        conditioning_8=None,
        confirmed_mask_1=None,
        confirmed_mask_2=None,
        confirmed_mask_3=None,
        confirmed_mask_4=None,
        confirmed_mask_5=None,
        confirmed_mask_6=None,
        confirmed_mask_7=None,
        confirmed_mask_8=None,
    ) -> io.NodeOutput:
        conditionings = [
            conditioning_1,
            conditioning_2,
            conditioning_3,
            conditioning_4,
            conditioning_5,
            conditioning_6,
            conditioning_7,
            conditioning_8,
        ]
        conditionings = [value for value in conditionings if value is not None]
        confirmed_masks = [
            confirmed_mask_1,
            confirmed_mask_2,
            confirmed_mask_3,
            confirmed_mask_4,
            confirmed_mask_5,
            confirmed_mask_6,
            confirmed_mask_7,
            confirmed_mask_8,
        ]
        frame_count, height, width, _channels = images.shape
        anchor_frames = _parse_forced_frame_indices(anchor_frames_json, int(frame_count))
        if not anchor_frames:
            raise ValueError("多锚点检测至少需要一个锚点帧")
        if len(anchor_frames) != len(conditionings):
            raise ValueError(
                f"锚点帧数量 {len(anchor_frames)} 与提示词数量 {len(conditionings)} 不一致"
            )
        try:
            raw_active_anchors = json.loads(active_anchors_json or "[]")
        except json.JSONDecodeError as error:
            raise ValueError(f"锚点启用状态不是合法 JSON: {error}") from error
        if not isinstance(raw_active_anchors, list):
            raise ValueError("锚点启用状态必须是数组")
        if raw_active_anchors:
            if len(raw_active_anchors) != len(anchor_frames):
                raise ValueError(
                    f"锚点帧数量 {len(anchor_frames)} 与启用状态数量 "
                    f"{len(raw_active_anchors)} 不一致"
                )
            if any(not isinstance(value, bool) for value in raw_active_anchors):
                raise ValueError("锚点启用状态只能是 true 或 false")
            active_anchors = [bool(value) for value in raw_active_anchors]
        else:
            active_anchors = [True for _ in anchor_frames]

        ordered_pairs = sorted(
            zip(anchor_frames, conditionings, active_anchors),
            key=lambda value: value[0],
        )
        anchor_frames = [value[0] for value in ordered_pairs]
        conditionings = [value[1] for value in ordered_pairs]
        active_anchors = [value[2] for value in ordered_pairs]
        key_indices = _anchored_keyframe_indices(
            images,
            int(interval),
            float(scene_threshold),
            anchor_frames_json,
        )
        if not key_indices:
            raise ValueError("视频没有可检测的关键帧")

        image_in = None
        if not bool(roi_enabled):
            image_in = comfy.utils.common_upscale(
                images[..., :3].movedim(-1, 1),
                1008,
                1008,
                "bilinear",
                crop="disabled",
            )
        comfy.model_management.load_model_gpu(model)
        device = comfy.model_management.get_torch_device()
        dtype = model.model.get_dtype()
        sam3_model = model.model.diffusion_model
        detector = sam3_model.detector
        vision_backbone = detector.backbone["vision_backbone"]
        text_resizer = detector.backbone["language_backbone"]["resizer"]
        prepared_prompts = []
        for conditioning in conditionings:
            prompts = _extract_text_prompts(conditioning, device, dtype)
            if not prompts:
                raise ValueError("锚点缺少 SAM3 文本提示词")
            prepared_prompts.append([
                (
                    text_resizer(text_embeddings),
                    text_mask.bool() if text_mask is not None else None,
                    max_detections,
                )
                for text_embeddings, text_mask, max_detections in prompts
            ])

        all_bbox_dicts = [[] for _ in key_indices]
        all_masks = [torch.zeros(height, width, device=device, dtype=torch.float32) for _ in key_indices]
        assignments = assign_keyframes_to_anchor_segments(
            key_indices=key_indices,
            anchor_frames=anchor_frames,
            shot_starts=_scene_shot_starts(images, float(scene_threshold)),
            active_anchors=active_anchors,
            reacquire_future_shots=reacquire_on_hard_cut,
        )
        if bool(roi_enabled):
            try:
                initial_boxes = json.loads(initial_boxes_json or "[]")
            except json.JSONDecodeError as error:
                raise ValueError(f"初始 ROI 不是合法 JSON: {error}") from error
            if not isinstance(initial_boxes, list):
                raise ValueError("初始 ROI 必须是数组")
            full_frame_box = {
                "x": 0,
                "y": 0,
                "width": int(width),
                "height": int(height),
            }
            initial_boxes = [
                value if isinstance(value, dict) else dict(full_frame_box)
                for value in initial_boxes
            ]
            if len(initial_boxes) < len(anchor_frames):
                initial_boxes.extend(
                    [dict(full_frame_box)] * (len(anchor_frames) - len(initial_boxes))
                )
            constraint_mode = str(roi_constraint_mode or "SEARCH_ROI").upper()
            if constraint_mode not in {"SEARCH_ROI", "HARD_BOUNDARY"}:
                raise ValueError("roi_constraint_mode 必须是 SEARCH_ROI 或 HARD_BOUNDARY")
            roi_masks, roi_bboxes = _execute_roi_keyframe_detection(
                model=model,
                sam3_model=sam3_model,
                vision_backbone=vision_backbone,
                images=images,
                key_indices=key_indices,
                assignments=assignments,
                anchor_frames=anchor_frames,
                active_anchors=active_anchors,
                prepared_prompts=prepared_prompts,
                confirmed_masks=confirmed_masks,
                initial_boxes=initial_boxes,
                threshold=float(threshold),
                refine_iterations=int(refine_iterations),
                device=device,
                dtype=dtype,
                height=int(height),
                width=int(width),
                roi_margin=float(roi_margin),
                roi_max_grace_frames=int(roi_max_grace_frames),
                roi_constraint_mode=constraint_mode,
                scene_threshold=float(scene_threshold),
                reacquire_on_hard_cut=reacquire_on_hard_cut,
            )
            return io.NodeOutput(roi_masks, roi_bboxes)
        assigned_key_positions = {
            key_position
            for anchor_assignments in assignments
            for key_position, _frame_index in anchor_assignments
        }
        pbar = comfy.utils.ProgressBar(len(key_indices))
        for key_position in range(len(key_indices)):
            if key_position not in assigned_key_positions:
                pbar.update(1)

        batch_size = max(1, min(int(frame_batch_size), len(key_indices)))
        scale = torch.tensor([width, height, width, height], device=device, dtype=dtype)
        for anchor_position, assigned in enumerate(assignments):
            prompts = prepared_prompts[anchor_position]
            for chunk_start in range(0, len(assigned), batch_size):
                chunk = assigned[chunk_start:chunk_start + batch_size]
                frame_indices = [frame_index for _key_position, frame_index in chunk]
                index_tensor = torch.tensor(frame_indices, device=image_in.device, dtype=torch.long)
                frames = image_in.index_select(0, index_tensor).to(device=device, dtype=dtype)
                trunk_out = vision_backbone.trunk(frames)
                prompt_results = []
                for text_embeddings, text_mask, max_detections in prompts:
                    result = detector.forward_from_trunk(trunk_out, text_embeddings, text_mask)
                    prompt_results.append((
                        result["boxes"] * scale,
                        result["scores"].sigmoid(),
                        F.interpolate(
                            result["masks"],
                            size=(height, width),
                            mode="bilinear",
                            align_corners=False,
                        ),
                        max_detections,
                    ))

                for local_index, (key_position, frame_index) in enumerate(chunk):
                    frame_bbox_dicts = []
                    frame_masks = []
                    for boxes, probabilities, masks, max_detections in prompt_results:
                        frame_boxes = boxes[local_index]
                        frame_probabilities = probabilities[local_index]
                        frame_masks_for_prompt = masks[local_index]
                        keep = frame_probabilities > float(threshold)
                        kept_boxes = frame_boxes[keep]
                        kept_probabilities = frame_probabilities[keep]
                        kept_masks = frame_masks_for_prompt[keep]
                        order = kept_probabilities.argsort(descending=True)[:max_detections]
                        kept_boxes = kept_boxes[order]
                        kept_probabilities = kept_probabilities[order]
                        kept_masks = kept_masks[order]
                        for box, score in zip(kept_boxes, kept_probabilities):
                            cpu_box = box.detach().cpu()
                            frame_bbox_dicts.append({
                                "x": float(cpu_box[0]),
                                "y": float(cpu_box[1]),
                                "width": float(cpu_box[2] - cpu_box[0]),
                                "height": float(cpu_box[3] - cpu_box[1]),
                                "score": float(score.detach().cpu()),
                            })
                        for coarse_mask, box in zip(kept_masks, kept_boxes):
                            frame_masks.append(_refine_mask(
                                sam3_model,
                                images[frame_index],
                                coarse_mask,
                                box,
                                height,
                                width,
                                device,
                                dtype,
                                int(refine_iterations),
                            ))
                    all_bbox_dicts[key_position] = frame_bbox_dicts
                    if frame_masks:
                        combined = torch.cat(frame_masks, dim=0)
                        all_masks[key_position] = (combined > 0).any(dim=0).float()
                    pbar.update(1)
                del trunk_out, prompt_results

        intermediate_device = comfy.model_management.intermediate_device()
        mask_out = torch.stack([mask.to(intermediate_device) for mask in all_masks])
        return io.NodeOutput(mask_out, all_bbox_dicts)


class SAM3InjectConfirmedKeyMasks(io.ComfyNode):
    """Replace sparse text-detected key masks with user-confirmed anchor masks."""

    @classmethod
    def define_schema(cls):
        inputs = [
            io.Image.Input("images", display_name="images"),
            io.Mask.Input("key_masks", display_name="key_masks"),
            io.Mask.Input("confirmed_mask_1", display_name="confirmed_mask_1"),
            io.String.Input(
                "anchor_frames_json",
                display_name="anchor_frames_json",
                default="[0]",
                multiline=True,
            ),
            io.Int.Input("interval", display_name="interval", default=6, min=1, max=120),
            io.Float.Input(
                "scene_threshold",
                display_name="scene_threshold",
                default=0.12,
                min=0.0,
                max=1.0,
                step=0.01,
            ),
        ]
        inputs.extend(
            io.Mask.Input(
                f"confirmed_mask_{index}",
                display_name=f"confirmed_mask_{index}",
                optional=True,
            )
            for index in range(2, 9)
        )
        return io.Schema(
            node_id="SAM3_InjectConfirmedKeyMasks",
            display_name="Inject Confirmed SAM3 Key Masks",
            category="image/detection",
            inputs=inputs,
            outputs=[io.Mask.Output("key_masks", display_name="key_masks")],
        )

    @classmethod
    def execute(
        cls,
        images,
        key_masks,
        confirmed_mask_1,
        anchor_frames_json="[0]",
        interval=6,
        scene_threshold=0.12,
        confirmed_mask_2=None,
        confirmed_mask_3=None,
        confirmed_mask_4=None,
        confirmed_mask_5=None,
        confirmed_mask_6=None,
        confirmed_mask_7=None,
        confirmed_mask_8=None,
    ) -> io.NodeOutput:
        confirmed_masks = [
            confirmed_mask_1,
            confirmed_mask_2,
            confirmed_mask_3,
            confirmed_mask_4,
            confirmed_mask_5,
            confirmed_mask_6,
            confirmed_mask_7,
            confirmed_mask_8,
        ]
        confirmed_masks = [mask for mask in confirmed_masks if mask is not None]
        frame_count = int(images.shape[0])
        anchor_frames = _parse_forced_frame_indices(anchor_frames_json, frame_count)
        if not anchor_frames:
            raise ValueError("确认遮罩至少需要一个锚点帧")
        if len(anchor_frames) != len(confirmed_masks):
            raise ValueError(
                f"锚点帧数量 {len(anchor_frames)} 与确认遮罩数量 {len(confirmed_masks)} 不一致"
            )
        key_indices = _anchored_keyframe_indices(
            images,
            int(interval),
            float(scene_threshold),
            anchor_frames_json,
        )
        if int(key_masks.shape[0]) != len(key_indices):
            raise ValueError(
                f"关键帧遮罩数量 {key_masks.shape[0]} 与预期关键帧数量 {len(key_indices)} 不一致"
            )
        index_to_position = {frame_index: position for position, frame_index in enumerate(key_indices)}
        result = key_masks.detach().float().clone()
        target_height, target_width = result.shape[-2:]
        for anchor_frame, confirmed_mask in zip(anchor_frames, confirmed_masks):
            key_position = index_to_position.get(anchor_frame)
            if key_position is None:
                raise ValueError(f"确认遮罩锚点帧 {anchor_frame} 不在关键帧序列中")
            current = confirmed_mask.detach().float()
            if current.ndim == 2:
                current = current.unsqueeze(0)
            if current.ndim != 3 or current.shape[0] < 1:
                raise ValueError("确认遮罩必须是 [帧数, 高, 宽] 格式")
            if current.shape[0] > 1:
                current = current.max(dim=0, keepdim=True).values
            if tuple(current.shape[-2:]) != (target_height, target_width):
                current = F.interpolate(
                    current.unsqueeze(1),
                    size=(target_height, target_width),
                    mode="bilinear",
                    align_corners=False,
                )[:, 0]
            result[key_position] = current[0].to(
                device=result.device,
                dtype=result.dtype,
            ).clamp(0.0, 1.0)
        return io.NodeOutput(result.to(device=key_masks.device, dtype=key_masks.dtype))


class SAM3SelectKeyframes(io.ComfyNode):
    """Select sparse frames for independent SAM3 detection; always includes the final frame."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3_SelectKeyframes",
            display_name="Select SAM3 Keyframes",
            category="image/detection",
            inputs=[
                io.Image.Input("images", display_name="images"),
                io.Int.Input("interval", display_name="interval", default=6, min=1, max=120),
                io.Float.Input(
                    "scene_threshold",
                    display_name="scene_threshold",
                    default=0.12,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                ),
            ],
            outputs=[io.Image.Output("keyframes", display_name="keyframes")],
        )

    @classmethod
    def execute(cls, images, interval=6, scene_threshold=0.12) -> io.NodeOutput:
        indices = _scene_keyframe_indices(images, int(interval), float(scene_threshold))
        if not indices:
            raise ValueError("视频没有可抽取的关键帧")
        index_tensor = torch.tensor(indices, device=images.device, dtype=torch.long)
        return io.NodeOutput(images.index_select(0, index_tensor))


class SAM3OpticalFlowMasks(io.ComfyNode):
    """Bidirectionally propagate sparse SAM3 masks between keyframes using dense optical flow."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3_OpticalFlowMasks",
            display_name="Interpolate SAM3 Masks with Optical Flow",
            category="image/detection",
            inputs=[
                io.Image.Input("images", display_name="images"),
                io.Mask.Input("key_masks", display_name="key_masks"),
                io.Int.Input("interval", display_name="interval", default=6, min=1, max=120),
                io.Float.Input("flow_scale", display_name="flow_scale", default=0.5, min=0.2, max=1.0, step=0.1),
                io.Float.Input(
                    "scene_threshold",
                    display_name="scene_threshold",
                    default=0.12,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                ),
                io.Float.Input(
                    "key_blend",
                    display_name="key_blend",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.1,
                    tooltip="关键帧软过渡: 光流传播 mask 与新检测 mask 的混合比例(0=硬切换)",
                ),
                io.Boolean.Input(
                    "anchor_guidance_enabled",
                    display_name="anchor_guidance_enabled",
                    default=False,
                    tooltip="用用户确认锚点的光流预测补全同一镜头内残缺的文本关键帧",
                ),
                io.Int.Input(
                    "anchor_grace_intervals",
                    display_name="anchor_grace_intervals",
                    default=3,
                    min=1,
                    max=12,
                    tooltip="确认锚点允许连续空检测使用光流补全的关键帧间隔数量",
                ),
                io.Boolean.Input(
                    "anchor_union_with_detection",
                    display_name="anchor_union_with_detection",
                    default=False,
                    tooltip="锚点 grace 内将当前检测与光流前景取并集，仅适合前景保留任务",
                ),
                io.Float.Input(
                    "anchor_visibility_color_threshold",
                    display_name="anchor_visibility_color_threshold",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.005,
                    tooltip="大于零时，锚点反向补全在目标色度失配后立即停止",
                ),
                io.String.Input(
                    "forced_frames_json",
                    display_name="forced_frames_json",
                    default="[]",
                    multiline=True,
                ),
            ],
            outputs=[io.Mask.Output("masks", display_name="masks")],
        )

    @classmethod
    def execute(
        cls,
        images,
        key_masks,
        interval=6,
        flow_scale=0.5,
        scene_threshold=0.12,
        key_blend=0.5,
        anchor_guidance_enabled=False,
        anchor_grace_intervals=3,
        anchor_union_with_detection=False,
        anchor_visibility_color_threshold=0.0,
        forced_frames_json="[]",
    ) -> io.NodeOutput:
        import cv2

        frame_count, height, width, _channels = images.shape
        indices = _anchored_keyframe_indices(
            images, int(interval), float(scene_threshold), forced_frames_json
        )
        forced_frames = set(_parse_forced_frame_indices(forced_frames_json, int(frame_count)))
        if int(key_masks.shape[0]) != len(indices):
            raise ValueError(
                f"关键帧遮罩数量 {key_masks.shape[0]} 与预期关键帧数量 {len(indices)} 不一致"
            )

        normalized_masks = key_masks.detach().float()
        if tuple(normalized_masks.shape[-2:]) != (height, width):
            normalized_masks = F.interpolate(
                normalized_masks.unsqueeze(1),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[:, 0]
        key_mask_np = normalized_masks.cpu().numpy().astype(np.float32)
        scale = max(0.2, min(1.0, float(flow_scale)))
        flow_width = max(32, int(round(width * scale)))
        flow_height = max(32, int(round(height * scale)))

        # Cache low-resolution grayscale frames once. The previous implementation
        # transferred the same MPS frame to CPU again in forward and backward passes.
        # Chunked transfer keeps peak memory bounded for 30-second 720p videos.
        gray_frames = np.empty(
            (int(frame_count), flow_height, flow_width),
            dtype=np.uint8,
        )
        transfer_batch_size = 8
        for batch_start in range(0, int(frame_count), transfer_batch_size):
            batch_end = min(int(frame_count), batch_start + transfer_batch_size)
            rgb_batch = images[batch_start:batch_end, ..., :3].detach().float().cpu().numpy()
            rgb_batch = (np.clip(rgb_batch, 0.0, 1.0) * 255.0).astype(np.uint8)
            for offset, rgb8 in enumerate(rgb_batch):
                if (flow_width, flow_height) != (width, height):
                    rgb8 = cv2.resize(
                        rgb8,
                        (flow_width, flow_height),
                        interpolation=cv2.INTER_AREA,
                    )
                gray_frames[batch_start + offset] = cv2.cvtColor(
                    rgb8,
                    cv2.COLOR_RGB2GRAY,
                )
            del rgb_batch

        def gray(frame_index: int) -> np.ndarray:
            return gray_frames[frame_index]

        grid_x, grid_y = np.meshgrid(
            np.arange(flow_width, dtype=np.float32),
            np.arange(flow_height, dtype=np.float32),
        )

        def warp(source_mask: np.ndarray, source_gray: np.ndarray, target_gray: np.ndarray) -> np.ndarray:
            # Backward flow tells each target pixel where to sample in the source frame.
            backward_flow = cv2.calcOpticalFlowFarneback(
                target_gray,
                source_gray,
                None,
                0.5,
                3,
                25,
                5,
                7,
                1.5,
                0,
            )
            source_small = source_mask
            if source_small.shape != (flow_height, flow_width):
                source_small = cv2.resize(
                    source_small,
                    (flow_width, flow_height),
                    interpolation=cv2.INTER_LINEAR,
                )
            warped_small = cv2.remap(
                source_small,
                grid_x + backward_flow[..., 0],
                grid_y + backward_flow[..., 1],
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            if (flow_width, flow_height) == (width, height):
                return np.clip(warped_small, 0.0, 1.0)
            return np.clip(
                cv2.resize(warped_small, (width, height), interpolation=cv2.INTER_LINEAR),
                0.0,
                1.0,
            )

        output = np.zeros((frame_count, height, width), dtype=np.float32)
        stabilized_keys = key_mask_np.copy()
        output[indices[0]] = stabilized_keys[0]
        shot_starts = set(_scene_shot_starts(images, float(scene_threshold)))
        reset_frames = forced_frames | shot_starts
        use_anchor_guidance = bool(anchor_guidance_enabled) and bool(forced_frames)
        active_anchor_frame = select_active_tracking_anchor(
            None,
            indices[0],
            mask_nonempty=bool(stabilized_keys[0].any()),
            hard_cut=indices[0] in shot_starts,
        )
        empty_grace_frames = max(1, int(interval)) * max(
            1,
            min(12, int(anchor_grace_intervals)),
        )


        def propagate_forward(
            source_mask: np.ndarray,
            left_frame: int,
            right_frame: int,
        ) -> dict[int, np.ndarray]:
            propagated: dict[int, np.ndarray] = {}
            current = source_mask
            source_gray = gray(left_frame)
            for frame_index in range(left_frame + 1, right_frame + 1):
                target_gray = gray(frame_index)
                current = warp(current, source_gray, target_gray)
                propagated[frame_index] = current
                source_gray = target_gray
            return propagated

        for segment_index in range(len(indices) - 1):
            left = indices[segment_index]
            right = indices[segment_index + 1]
            right_detected = key_mask_np[segment_index + 1]
            if right <= left:
                stabilized_keys[segment_index + 1] = right_detected
                output[right] = right_detected
                continue

            left_mask = stabilized_keys[segment_index]
            if not left_mask.any():
                stabilized_keys[segment_index + 1] = right_detected
                output[right] = right_detected
                active_anchor_frame = select_active_tracking_anchor(
                    active_anchor_frame,
                    right,
                    mask_nonempty=bool(right_detected.any()),
                    hard_cut=right in shot_starts,
                )
                continue

            if not right_detected.any():
                within_anchor_grace = (
                    use_anchor_guidance
                    and active_anchor_frame is not None
                    and right not in shot_starts
                    and right - active_anchor_frame <= empty_grace_frames
                )
                if within_anchor_grace:
                    forward = propagate_forward(left_mask, left, right)
                    stabilized_keys[segment_index + 1] = forward[right]
                    for frame_index in range(left + 1, right + 1):
                        output[frame_index] = forward[frame_index]
                    continue
                stabilized_keys[segment_index + 1] = right_detected
                output[right] = right_detected
                active_anchor_frame = select_active_tracking_anchor(
                    active_anchor_frame,
                    right,
                    mask_nonempty=False,
                    hard_cut=right in shot_starts,
                )
                continue

            forward = propagate_forward(left_mask, left, right)
            within_anchor_grace = (
                use_anchor_guidance
                and active_anchor_frame is not None
                and right not in shot_starts
                and right - active_anchor_frame <= empty_grace_frames
            )

            # A non-empty mask already passed SAM3_GateKeyMasks, so the current
            # frame detection is authoritative. Optical flow may bridge empty
            # detections above, but must never be unioned into a valid keyframe:
            # doing so fills moving gaps (for example between legs) and carries
            # detached ghost components from older poses.
            stabilized_keys[segment_index + 1] = select_keyframe_mask(
                right_detected,
                forward[right],
                preserve_propagated=(
                    bool(anchor_union_with_detection) and within_anchor_grace
                ),
            )
            right_mask = stabilized_keys[segment_index + 1]
            active_anchor_frame = select_active_tracking_anchor(
                active_anchor_frame,
                right,
                mask_nonempty=bool(right_mask.any()),
                hard_cut=right in shot_starts,
            )

            # 只有真正切镜才禁止反向混入；同镜头用户锚点可反向约束前一段。
            if right in shot_starts:
                for frame_index in range(left + 1, right):
                    output[frame_index] = forward[frame_index]
                output[right] = right_mask
                continue

            backward = {}
            current_mask = right_mask
            source_gray = gray(right)
            for frame_index in range(right - 1, left, -1):
                target_gray = gray(frame_index)
                current_mask = warp(current_mask, source_gray, target_gray)
                backward[frame_index] = current_mask
                source_gray = target_gray

            segment_length = right - left
            for frame_index in range(left + 1, right):
                alpha = (frame_index - left) / segment_length
                output[frame_index] = np.clip(
                    forward[frame_index] * (1.0 - alpha) + backward[frame_index] * alpha,
                    0.0,
                    1.0,
                )

            # Keyframes always retain the accepted detector topology. The
            # key_blend input remains in the schema for stored-workflow compatibility.
            output[right] = right_mask

        # A user may confirm a subject after frame 0 because it is absent, occluded,
        # or unclear at the beginning. When all earlier detector masks are empty,
        # propagate that confirmed mask backwards only inside the same shot. This
        # mirrors the forward empty-mask grace window and prevents the next shot's
        # subject from leaking backwards across a hard cut. Existing non-empty masks
        # are treated as reliable and are never overwritten.
        if use_anchor_guidance:
            index_to_position = {frame_index: position for position, frame_index in enumerate(indices)}
            ordered_shot_starts = sorted(shot_starts)
            for anchor_frame in sorted(forced_frames):
                anchor_position = index_to_position.get(anchor_frame)
                if anchor_position is None:
                    continue
                anchor_mask = key_mask_np[anchor_position]
                if not anchor_mask.any():
                    continue

                shot_start = 0
                for candidate in ordered_shot_starts:
                    if candidate > anchor_frame:
                        break
                    shot_start = candidate
                start_frame = max(shot_start, anchor_frame - empty_grace_frames)

                current_mask = anchor_mask
                reference_image = None
                visibility_threshold = float(anchor_visibility_color_threshold)
                if visibility_threshold > 0:
                    reference_image = (
                        images[anchor_frame, ..., :3]
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                    )
                source_gray = gray(anchor_frame)
                for frame_index in range(anchor_frame - 1, start_frame - 1, -1):
                    target_gray = gray(frame_index)
                    current_mask = warp(current_mask, source_gray, target_gray)
                    if reference_image is not None:
                        current_image = (
                            images[frame_index, ..., :3]
                            .detach()
                            .float()
                            .cpu()
                            .numpy()
                        )
                        appearance_distance = masked_color_signature_distance(
                            reference_image,
                            anchor_mask,
                            current_image,
                            current_mask,
                        )
                        if not anchor_visibility_allows(
                            appearance_distance,
                            visibility_threshold,
                        ):
                            break
                    if not output[frame_index].any():
                        output[frame_index] = current_mask
                    source_gray = target_gray

                # The confirmed anchor is ground truth and must remain byte-for-byte
                # equivalent to the injected key mask after both propagation passes.
                output[anchor_frame] = anchor_mask

        result = torch.from_numpy(output).to(device=key_masks.device, dtype=key_masks.dtype)
        return io.NodeOutput(result)


class SAM3GateKeyMasks(io.ComfyNode):
    """Filter unreliable per-keyframe masks before optical-flow propagation.

    Rules per keyframe (evaluated in keyframe order):
    - empty mask stays empty (target genuinely absent → downstream segment is emptied);
    - mask covering >= max_full_frac of the frame is dropped (a subject mask never fills
      the frame — this kills full-green transition frames);
    - mask whose area deviates from the rolling median of recent accepted keyframes by
      more than max_area_ratio is dropped (kills semantic drift like mask jumping to a
      hand/egg/UI card). History resets at every scene cut, so new shots may legitimately
      have a different mask size.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3_GateKeyMasks",
            display_name="Gate SAM3 Keyframe Masks",
            category="image/detection",
            inputs=[
                io.Image.Input("images", display_name="images"),
                io.Mask.Input("key_masks", display_name="key_masks"),
                io.Int.Input("interval", display_name="interval", default=6, min=1, max=120),
                io.Float.Input(
                    "scene_threshold",
                    display_name="scene_threshold",
                    default=0.12,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                ),
                io.Int.Input("window", display_name="window", default=4, min=1, max=16),
                io.Float.Input(
                    "max_area_ratio",
                    display_name="max_area_ratio",
                    default=3.0,
                    min=1.0,
                    max=20.0,
                    step=0.5,
                ),
                io.Float.Input(
                    "max_full_frac",
                    display_name="max_full_frac",
                    default=0.6,
                    min=0.1,
                    max=1.0,
                    step=0.05,
                ),
                io.String.Input(
                    "forced_frames_json",
                    display_name="forced_frames_json",
                    default="[]",
                    multiline=True,
                ),
            ],
            outputs=[io.Mask.Output("masks", display_name="masks")],
        )

    @classmethod
    def execute(
        cls,
        images,
        key_masks,
        interval=6,
        scene_threshold=0.12,
        window=4,
        max_area_ratio=3.0,
        max_full_frac=0.6,
        forced_frames_json="[]",
    ) -> io.NodeOutput:
        indices = _anchored_keyframe_indices(
            images, int(interval), float(scene_threshold), forced_frames_json
        )
        if int(key_masks.shape[0]) != len(indices):
            raise ValueError(
                f"关键帧遮罩数量 {key_masks.shape[0]} 与预期关键帧数量 {len(indices)} 不一致"
            )
        shot_starts = set(_scene_shot_starts(images, float(scene_threshold)))
        forced_starts = set(_parse_forced_frame_indices(forced_frames_json, int(images.shape[0])))
        masks_np = key_masks.detach().float().cpu().numpy()
        areas = (
            masks_np.reshape(len(indices), -1).mean(axis=1)
            if indices
            else np.zeros(0, dtype=np.float32)
        )
        gated = np.zeros_like(masks_np)
        history: list[float] = []
        ratio = max(1.0, float(max_area_ratio))
        full_frac = max(0.1, min(1.0, float(max_full_frac)))
        win = max(1, int(window))
        for pos, frame_index in enumerate(indices):
            area = float(areas[pos])
            if frame_index == 0 or frame_index in shot_starts or frame_index in forced_starts:
                history = []
            accept = 1e-6 < area < full_frac
            if accept and history:
                median = float(np.median(history[-win:]))
                if median > 0 and (area > ratio * median or area < median / ratio):
                    accept = False
            if accept:
                gated[pos] = masks_np[pos]
                history.append(area)
        result = torch.from_numpy(gated).to(device=key_masks.device, dtype=key_masks.dtype)
        return io.NodeOutput(result)


class SAM3UnionMasks(io.ComfyNode):
    """Element-wise union (max) of two mask batches — merges per-part detections into one stable mask."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3_UnionMasks",
            display_name="Union SAM3 Masks",
            category="image/detection",
            inputs=[
                io.Mask.Input("mask_a", display_name="mask_a"),
                io.Mask.Input("mask_b", display_name="mask_b"),
            ],
            outputs=[io.Mask.Output("mask", display_name="mask")],
        )

    @classmethod
    def execute(cls, mask_a, mask_b) -> io.NodeOutput:
        a = mask_a.detach().float()
        b = mask_b.detach().float()
        if tuple(a.shape[-2:]) != tuple(b.shape[-2:]):
            b = F.interpolate(
                b.unsqueeze(1), size=tuple(a.shape[-2:]),
                mode="bilinear", align_corners=False,
            )[:, 0]
        count = max(int(a.shape[0]), int(b.shape[0]))
        if int(a.shape[0]) != count:
            a = a.expand(count, -1, -1)
        if int(b.shape[0]) != count:
            b = b.expand(count, -1, -1)
        return io.NodeOutput(torch.maximum(a, b))


class SAM3TemporalSmoothMasks(io.ComfyNode):
    """Per-shot temporal median over mask probabilities (±radius frames, never across cuts).

    Suppresses single-frame flicker and edge boiling without touching shot boundaries:
    a mask that appears/disappears for only one frame within the window is voted out.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3_TemporalSmoothMasks",
            display_name="Temporal Smooth SAM3 Masks",
            category="image/detection",
            inputs=[
                io.Image.Input("images", display_name="images"),
                io.Mask.Input("masks", display_name="masks"),
                io.Int.Input("radius", display_name="radius", default=1, min=1, max=4),
                io.Int.Input("interval", display_name="interval", default=12, min=1, max=120),
                io.String.Input(
                    "forced_frames_json",
                    display_name="forced_frames_json",
                    default="[]",
                    multiline=True,
                ),
                io.Float.Input(
                    "scene_threshold",
                    display_name="scene_threshold",
                    default=0.12,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                ),
            ],
            outputs=[io.Mask.Output("masks", display_name="masks")],
        )

    @classmethod
    def execute(
        cls,
        images,
        masks,
        radius=1,
        interval=12,
        forced_frames_json="[]",
        scene_threshold=0.12,
    ) -> io.NodeOutput:
        frame_count = int(masks.shape[0])
        if frame_count <= 2:
            return io.NodeOutput(masks)
        shot_starts = _scene_shot_starts(images, float(scene_threshold))
        preserve_indices = _anchored_keyframe_indices(
            images,
            int(interval),
            float(scene_threshold),
            forced_frames_json,
        )
        masks_np = masks.detach().float().cpu().numpy()
        smoothed = temporal_median_preserving_indices(
            masks_np,
            shot_starts=shot_starts,
            radius=int(radius),
            preserve_indices=preserve_indices,
        )
        result = torch.from_numpy(smoothed).to(device=masks.device, dtype=masks.dtype)
        return io.NodeOutput(result)


class SAM3MaskTrackBoxes(io.ComfyNode):
    """Derive a per-frame tracking ROI from key masks.

    Non-empty mask → its bounding box expanded by margin; empty mask → carry the
    previous box forward (falls back to initial_box_json, then full frame). Feeds
    per-frame crop/detect loops so small targets are re-detected zoomed-in instead
    of being lost in the full frame.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3_MaskTrackBoxes",
            display_name="SAM3 Mask Track Boxes",
            category="image/detection",
            inputs=[
                io.Mask.Input("key_masks", display_name="key_masks"),
                io.String.Input("initial_box_json", display_name="initial_box_json",
                                default="[]", multiline=True),
                io.Float.Input("margin", display_name="margin", default=0.6,
                               min=0.0, max=4.0, step=0.1),
            ],
            outputs=[io.String.Output("boxes_json", display_name="boxes_json")],
        )

    @classmethod
    def execute(cls, key_masks, initial_box_json="[]", margin=0.6) -> io.NodeOutput:
        import json

        masks = key_masks.detach().float().cpu().numpy()
        count, height, width = masks.shape
        fallback = None
        try:
            parsed = json.loads(initial_box_json or "[]")
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                fallback = parsed[0]
        except json.JSONDecodeError:
            fallback = None
        if fallback is None:
            fallback = {"x": 0, "y": 0, "width": int(width), "height": int(height)}
        grow = 1.0 + max(0.0, float(margin))
        boxes = []
        last = None
        for index in range(count):
            ys, xs = np.nonzero(masks[index] > 0.5)
            if len(xs):
                left, right = int(xs.min()), int(xs.max()) + 1
                top, bottom = int(ys.min()), int(ys.max()) + 1
                center_x, center_y = (left + right) / 2.0, (top + bottom) / 2.0
                box_w = min(float(width), (right - left) * grow)
                box_h = min(float(height), (bottom - top) * grow)
                box = {
                    "x": max(0.0, center_x - box_w / 2.0),
                    "y": max(0.0, center_y - box_h / 2.0),
                    "width": box_w,
                    "height": box_h,
                }
                last = box
                boxes.append(box)
            else:
                boxes.append(dict(last) if last else dict(fallback))
        return io.NodeOutput(json.dumps(boxes))


def _parse_track_boxes(track_boxes_json: str):
    import json

    if not track_boxes_json:
        return None
    try:
        boxes = json.loads(track_boxes_json)
    except json.JSONDecodeError:
        return None
    if isinstance(boxes, list) and boxes and isinstance(boxes[0], dict):
        return boxes
    return None


class SAM3CropImagesToBox(io.ComfyNode):
    """Crop a fixed ROI from every image/video frame and upscale it for small-object segmentation."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3_CropImagesToBox",
            display_name="Crop Images to SAM3 Box",
            category="image/detection",
            inputs=[
                io.Image.Input("images", display_name="images"),
                io.String.Input("boxes_json", display_name="boxes_json", default="[]", multiline=True),
                io.Int.Input("target_width", display_name="target_width", default=940, min=64, max=2048),
                io.Int.Input("target_height", display_name="target_height", default=1400, min=64, max=2048),
                io.String.Input("track_boxes_json", display_name="track_boxes_json",
                                default="", multiline=True, optional=True),
            ],
            outputs=[io.Image.Output("cropped_images", display_name="cropped_images")],
        )

    @classmethod
    def execute(cls, images, boxes_json="[]", target_width=940, target_height=1400,
                track_boxes_json="") -> io.NodeOutput:
        _batch, height, width, _channels = images.shape
        track_boxes = _parse_track_boxes(track_boxes_json)
        if track_boxes:
            if len(track_boxes) != int(images.shape[0]):
                raise ValueError(
                    f"跟踪框数量 {len(track_boxes)} 与帧数 {images.shape[0]} 不一致"
                )
            frames = []
            for index, box in enumerate(track_boxes):
                left, top, right, bottom = _box_edges(box, width, height)
                if right - left <= 0 or bottom - top <= 0:
                    raise ValueError("跟踪框没有覆盖有效图像区域")
                crop = images[index:index + 1, top:bottom, left:right, :3]
                frames.append(F.interpolate(
                    crop.permute(0, 3, 1, 2),
                    size=(int(target_height), int(target_width)),
                    mode="bilinear",
                    align_corners=False,
                ))
            return io.NodeOutput(torch.cat(frames).permute(0, 2, 3, 1))
        box = _first_box(boxes_json)
        left, top, right, bottom = _box_edges(box, width, height)
        cropped = images[:, top:bottom, left:right, :]
        if cropped.shape[1] == 0 or cropped.shape[2] == 0:
            raise ValueError("裁剪框没有覆盖有效图像区域")
        resized = F.interpolate(
            cropped[..., :3].permute(0, 3, 1, 2),
            size=(int(target_height), int(target_width)),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)
        return io.NodeOutput(resized)


class SAM3PasteMaskToCanvas(io.ComfyNode):
    """Resize cropped-frame masks and paste them back into the full video canvas."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3_PasteMaskToCanvas",
            display_name="Paste SAM3 Mask to Canvas",
            category="image/detection",
            inputs=[
                io.Mask.Input("mask", display_name="mask"),
                io.String.Input("boxes_json", display_name="boxes_json", default="[]", multiline=True),
                io.Int.Input("canvas_width", display_name="canvas_width", default=720, min=64, max=8192),
                io.Int.Input("canvas_height", display_name="canvas_height", default=1280, min=64, max=8192),
                io.String.Input("track_boxes_json", display_name="track_boxes_json",
                                default="", multiline=True, optional=True),
            ],
            outputs=[io.Mask.Output("canvas_mask", display_name="canvas_mask")],
        )

    @classmethod
    def execute(cls, mask, boxes_json="[]", canvas_width=720, canvas_height=1280,
                track_boxes_json="") -> io.NodeOutput:
        width = int(canvas_width)
        height = int(canvas_height)
        track_boxes = _parse_track_boxes(track_boxes_json)
        canvas = torch.zeros(
            (mask.shape[0], height, width),
            device=mask.device,
            dtype=mask.dtype,
        )
        if track_boxes:
            if len(track_boxes) != int(mask.shape[0]):
                raise ValueError(
                    f"跟踪框数量 {len(track_boxes)} 与 mask 帧数 {mask.shape[0]} 不一致"
                )
            for index, box in enumerate(track_boxes):
                left, top, right, bottom = _box_edges(box, width, height)
                if right - left <= 0 or bottom - top <= 0:
                    continue
                resized = F.interpolate(
                    mask[index:index + 1].detach().float().unsqueeze(1),
                    size=(bottom - top, right - left),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
                canvas[index, top:bottom, left:right] = resized.to(
                    device=mask.device, dtype=mask.dtype)
            return io.NodeOutput(canvas)
        box = _first_box(boxes_json)
        left, top, right, bottom = _box_edges(box, width, height)
        resized = F.interpolate(
            mask.detach().float().unsqueeze(1),
            size=(bottom - top, right - left),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        canvas[:, top:bottom, left:right] = resized.to(device=mask.device, dtype=mask.dtype)
        return io.NodeOutput(canvas)


def _first_box(boxes_json: str) -> dict:
    import json

    try:
        boxes = json.loads(boxes_json or "[]")
    except json.JSONDecodeError as error:
        raise ValueError(f"裁剪框数据不是合法 JSON: {error}") from error
    if not isinstance(boxes, list) or not boxes or not isinstance(boxes[0], dict):
        raise ValueError("裁剪框不能为空")
    return boxes[0]


def _box_edges(box: dict, width: int, height: int) -> tuple[int, int, int, int]:
    try:
        left = max(0, min(width - 1, int(round(float(box["x"])))))
        top = max(0, min(height - 1, int(round(float(box["y"])))))
        right = max(left + 1, min(width, int(round(float(box["x"]) + float(box["width"])))))
        bottom = max(top + 1, min(height, int(round(float(box["y"]) + float(box["height"])))))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("裁剪框缺少 x/y/width/height") from error
    return left, top, right, bottom


class SAM3RefineMaskWithPoints(io.ComfyNode):
    """Use point prompts to refine an existing text/box mask instead of unioning masks."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3_RefineMaskWithPoints",
            display_name="Refine SAM3 Mask With Points",
            category="image/detection",
            inputs=[
                io.Model.Input("model", display_name="model"),
                io.Image.Input("image", display_name="image"),
                io.Mask.Input("base_mask", display_name="base_mask"),
                io.String.Input(
                    "positive_coords",
                    display_name="positive_coords",
                    default="[]",
                    multiline=True,
                    tooltip="正点 JSON [{\"x\": int, \"y\": int}, ...]，使用原图像素坐标",
                ),
                io.String.Input(
                    "negative_coords",
                    display_name="negative_coords",
                    default="[]",
                    multiline=True,
                    tooltip="负点 JSON [{\"x\": int, \"y\": int}, ...]，使用原图像素坐标",
                ),
                io.Int.Input(
                    "refine_iterations",
                    display_name="refine_iterations",
                    default=2,
                    min=1,
                    max=5,
                    step=1,
                ),
                io.Float.Input(
                    "mask_strength",
                    display_name="mask_strength",
                    default=2.0,
                    min=0.1,
                    max=20.0,
                    step=0.1,
                    tooltip="已有 mask 对交互式解码器的先验强度；越低越服从点提示",
                ),
            ],
            outputs=[io.Mask.Output("mask")],
        )

    @classmethod
    def execute(
        cls,
        model,
        image,
        base_mask,
        positive_coords="[]",
        negative_coords="[]",
        refine_iterations=2,
        mask_strength=2.0,
    ) -> io.NodeOutput:
        import json

        import comfy.model_management
        import comfy.utils

        if image.ndim != 4 or image.shape[-1] < 3:
            raise ValueError("image 必须是 [帧数, 高, 宽, 通道] 图像批次")
        if base_mask.ndim == 2:
            base_mask = base_mask.unsqueeze(0)
        if base_mask.ndim != 3:
            raise ValueError("base_mask 必须是 [帧数, 高, 宽] mask 批次")

        frame_count, height, width, _channels = image.shape
        if base_mask.shape[0] not in (1, frame_count):
            raise ValueError(f"base_mask 帧数 {base_mask.shape[0]} 与 image 帧数 {frame_count} 不一致")

        try:
            positive = json.loads(positive_coords or "[]")
            negative = json.loads(negative_coords or "[]")
        except json.JSONDecodeError as error:
            raise ValueError(f"点选坐标不是合法 JSON: {error}") from error
        if not isinstance(positive, list) or not isinstance(negative, list):
            raise ValueError("positive_coords/negative_coords 必须是数组")
        points = positive + negative
        for point in points:
            if not isinstance(point, dict) or "x" not in point or "y" not in point:
                raise ValueError("点坐标必须是包含 x/y 的对象")

        # SAM3 decoder 接收 1008x1008 图像、点坐标和 mask logits；这里把已有二值 mask
        # 转成强 logits，让点击只修正局部边界，而不是把文本 mask 与点 mask 做并集。
        image_in = comfy.utils.common_upscale(
            image[..., :3].movedim(-1, 1), 1008, 1008, "bilinear", crop="disabled"
        )
        comfy.model_management.load_model_gpu(model)
        device = comfy.model_management.get_torch_device()
        dtype = model.model.get_dtype()
        sam3_model = model.model.diffusion_model
        point_inputs = None
        if points:
            coords = [[float(point["x"]) / width * 1008.0,
                       float(point["y"]) / height * 1008.0] for point in points]
            labels = [1] * len(positive) + [0] * len(negative)
            point_inputs = {
                "point_coords": torch.tensor([coords], dtype=dtype, device=device),
                "point_labels": torch.tensor([labels], dtype=torch.int32, device=device),
            }

        output_masks = []
        for index in range(frame_count):
            frame = image_in[index:index + 1].to(device=device, dtype=dtype)
            source_mask = base_mask[0 if base_mask.shape[0] == 1 else index]
            source_mask = F.interpolate(
                source_mask.unsqueeze(0).unsqueeze(0).float().to(device=device),
                size=(1008, 1008),
                mode="bilinear",
                align_corners=False,
            )
            mask_inputs = ((source_mask * 2.0 - 1.0) * float(mask_strength)).to(dtype=dtype)
            refined = sam3_model.forward_segment(
                frame,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
            )
            for _ in range(max(0, int(refine_iterations) - 1)):
                refined = sam3_model.forward_segment(
                    frame,
                    point_inputs=point_inputs,
                    mask_inputs=refined,
                )
            if refined.ndim == 3:
                refined = refined.unsqueeze(1)
            candidates = F.interpolate(
                refined,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[0]
            if candidates.shape[0] > 1:
                base_binary = source_mask[0, 0] > 0
                base_binary = F.interpolate(
                    base_binary.float().unsqueeze(0).unsqueeze(0),
                    size=(height, width),
                    mode="nearest",
                )[0, 0] > 0.5
                candidate_binary = candidates > 0
                overlap = (candidate_binary & base_binary).float().sum(dim=(-2, -1))
                union = (candidate_binary | base_binary).float().sum(dim=(-2, -1)).clamp_min(1.0)
                scores = overlap / union
                selected = candidates[scores.argmax()]
            else:
                selected = candidates[0]
            output_masks.append((selected > 0).float().cpu())

        masks = torch.stack(output_masks, dim=0)
        return io.NodeOutput(masks.to(comfy.model_management.intermediate_device()))


class SAM3GreenScreenImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3_GreenScreenImage",
            display_name="SAM3 Green Screen Image",
            category="image/detection",
            inputs=[
                io.Image.Input("image", display_name="image"),
                io.Mask.Input("edit_mask", display_name="edit_mask"),
                io.String.Input("green_hex", display_name="green_hex", default="#00B140"),
            ],
            outputs=[io.Image.Output("preview", display_name="preview")],
        )

    @classmethod
    def execute(cls, image, edit_mask, green_hex="#00B140") -> io.NodeOutput:
        frame_count, height, width, _channels = image.shape
        mask = edit_mask.detach().float()
        if tuple(mask.shape[-2:]) != (height, width):
            mask = F.interpolate(mask.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False)[:, 0]
        if mask.shape[0] == 1 and frame_count > 1:
            mask = mask.expand(frame_count, -1, -1)
        green = torch.tensor(_parse_hex_color(green_hex), device=image.device, dtype=image.dtype).view(1, 1, 1, 3)
        alpha = mask.to(device=image.device, dtype=image.dtype).clamp(0.0, 1.0).unsqueeze(-1)
        return io.NodeOutput((image[..., :3] * (1.0 - alpha) + green * alpha).clamp(0.0, 1.0))


class SAM3GreenScreenVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3_GreenScreenVideo",
            display_name="SAM3 Green Screen Video",
            category="image/detection",
            inputs=[
                io.Image.Input("images", display_name="images"),
                io.Mask.Input("masks", display_name="masks"),
                io.Float.Input("fps", display_name="fps", default=24.0, min=1.0, max=120.0, step=1.0),
                io.String.Input("green_hex", display_name="green_hex", default="#00B140"),
                io.Int.Input("crf", display_name="crf", default=18, min=0, max=35),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images, masks, fps=24.0, green_hex="#00B140", crf=18) -> io.NodeOutput:
        frame_count, height, width, _channels = images.shape
        if masks.shape[0] not in (1, frame_count):
            raise ValueError(f"mask 帧数 {masks.shape[0]} 与视频帧数 {frame_count} 不一致")
        ffmpeg_bin = _resolve_ffmpeg_bin()

        filename = f"sam3_green_{uuid.uuid4().hex[:10]}.mp4"
        output_path = Path(folder_paths.get_temp_directory()) / filename
        green = np.asarray(_parse_hex_color(green_hex), dtype=np.float32).reshape(1, 1, 3)
        command = [
            str(ffmpeg_bin), "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{int(width)}x{int(height)}",
            "-r", str(max(1.0, float(fps))), "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "medium",
            "-crf", str(int(crf)), "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(output_path),
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            assert process.stdin is not None
            for index in range(frame_count):
                frame = images[index, ..., :3].detach().float().cpu().numpy()
                mask = masks[0 if masks.shape[0] == 1 else index]
                if tuple(mask.shape[-2:]) != (height, width):
                    mask = F.interpolate(
                        mask.unsqueeze(0).unsqueeze(0).float(),
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0]
                alpha = mask.detach().float().cpu().numpy().clip(0.0, 1.0)[..., None]
                composed = (frame * (1.0 - alpha) + green * alpha).clip(0.0, 1.0)
                process.stdin.write((composed * 255.0).astype(np.uint8).tobytes())
            process.stdin.close()
            stderr = process.stderr.read() if process.stderr is not None else b""
            return_code = process.wait()
        except Exception:
            process.kill()
            process.wait()
            raise
        if return_code != 0:
            output_path.unlink(missing_ok=True)
            raise ValueError(f"FFmpeg 视频编码失败: {stderr.decode('utf-8', 'replace')[-2000:]}")

        return io.NodeOutput(ui=ui.PreviewVideo([
            ui.SavedResult(filename, "", io.FolderType.temp)
        ]))


def _seed_identity_object_indices(
    frames: torch.Tensor,
    track_data: dict,
    *,
    seed_count: int,
    max_distance: float,
) -> str:
    """按种子颜色签名过滤 spawn 对象，返回 TrackToMask 的 object_indices。

    逐对象流式解包（每对象只取首个非空帧），内存 O(1)——全量 stack 会在
    长视频 × 多对象场景把整机内存打爆（1100 帧 × 40 对象 ≈ 14GB，实测触发内核 OOM）。
    """

    from comfy.ldm.sam3.tracker import unpack_masks

    packed = track_data["packed_masks"]
    frame_count = int(packed.shape[0])
    object_count = int(packed.shape[1])
    if object_count <= seed_count:
        return ""

    def first_visible(object_index: int) -> tuple[int, torch.Tensor] | None:
        for frame_index in range(frame_count):
            mask = unpack_masks(packed[frame_index:frame_index + 1, object_index])[0].float()
            if float(mask.sum().item()) > 0:
                return frame_index, mask.cpu()
        return None

    def frame_small(frame_index: int, height: int, width: int) -> torch.Tensor:
        return F.interpolate(
            frames[frame_index:frame_index + 1].movedim(-1, 1).float().cpu(),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).movedim(1, -1)[0]

    reference = first_visible(0)
    if reference is None:
        return ""
    reference_frame, reference_mask = reference
    mask_height, mask_width = reference_mask.shape[-2], reference_mask.shape[-1]
    reference_image = frame_small(reference_frame, mask_height, mask_width)

    kept = list(range(seed_count))
    for object_index in range(seed_count, object_count):
        visible = first_visible(object_index)
        if visible is None:
            continue
        frame_index, object_mask = visible
        distance = masked_color_signature_distance(
            reference_image.numpy(),
            reference_mask.numpy(),
            frame_small(frame_index, mask_height, mask_width).numpy(),
            object_mask.numpy(),
        )
        if distance is None or distance <= max_distance:
            kept.append(object_index)
    if len(kept) == object_count:
        return ""
    dropped = object_count - len(kept)
    print(f"[SAM3_TrackVideoMasks] 身份过滤剔除 {dropped}/{object_count - seed_count} 个相似对象")
    return ",".join(str(index) for index in kept)


def _resize_mask_batch(masks: torch.Tensor, height: int, width: int) -> torch.Tensor:
    masks = masks.to(dtype=torch.float32, device="cpu")
    if masks.shape[-2] == height and masks.shape[-1] == width:
        return masks
    return F.interpolate(
        masks.unsqueeze(1),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[:, 0]


class SAM3TrackVideoMasks(io.ComfyNode):
    """完整视频遮罩追踪：按 anchor 分段调用 ComfyUI core SAM3.1 tracker。

    每段以 anchor 确认遮罩为种子（首个 anchor 之前的帧倒放回溯），
    文本 conditioning 让 detector 在段内每 detect_interval 帧重检测新实例。
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3_TrackVideoMasks",
            display_name="SAM3 Track Video Masks",
            category="sam3_demo",
            inputs=[
                io.Image.Input("images", display_name="images"),
                io.Model.Input("model", display_name="model"),
                io.Conditioning.Input("conditioning", optional=True),
                io.Mask.Input(
                    "anchor_masks",
                    optional=True,
                    tooltip="每个 anchor 一张确认遮罩，批次顺序与 anchor_frames_json 对应",
                ),
                io.String.Input("anchor_frames_json", default="[]"),
                io.Float.Input(
                    "detection_threshold",
                    default=0.35,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                ),
                io.Float.Input(
                    "seeded_spawn_threshold",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="有 anchor 种子的分段用此阈值抑制文本新增实例；0=沿用 detection_threshold",
                ),
                io.Float.Input(
                    "seed_identity_max_distance",
                    default=0.0,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    tooltip="单主体身份过滤：spawn 对象与种子的颜色签名距离超过此值即剔除；0=关闭",
                ),
                io.Int.Input("max_objects", default=0, min=0, max=64),
                io.Int.Input("detect_interval", default=1, min=1),
                io.Boolean.Input("override_anchor_frames", default=True),
            ],
            outputs=[io.Mask.Output("masks")],
        )

    @classmethod
    def execute(
        cls,
        images,
        model,
        conditioning=None,
        anchor_masks=None,
        anchor_frames_json="[]",
        detection_threshold=0.35,
        seeded_spawn_threshold=0.0,
        seed_identity_max_distance=0.0,
        max_objects=0,
        detect_interval=1,
        override_anchor_frames=True,
    ) -> io.NodeOutput:
        from comfy_extras.nodes_sam3 import SAM3_TrackToMask, SAM3_VideoTrack

        frame_count = int(images.shape[0])
        height = int(images.shape[1])
        width = int(images.shape[2])
        anchor_frames = parse_anchor_frames(anchor_frames_json, frame_count)
        if anchor_masks is None:
            anchor_frames = []
        elif anchor_masks.shape[0] != len(anchor_frames):
            raise ValueError("anchor_masks 数量与 anchor_frames_json 不一致")
        if conditioning is None and anchor_masks is None:
            raise ValueError("conditioning 与 anchor_masks 至少提供一个")

        resized_anchor_masks = (
            _resize_mask_batch(anchor_masks, height, width)
            if anchor_masks is not None
            else None
        )
        segments = plan_tracking_segments(frame_count, anchor_frames)
        output = torch.zeros(frame_count, height, width, dtype=torch.float32)
        for segment in segments:
            frames = images[segment.start:segment.stop]
            if segment.reverse:
                frames = torch.flip(frames, dims=[0])
            seed = None
            if segment.anchor_index is not None and resized_anchor_masks is not None:
                candidate = resized_anchor_masks[
                    segment.anchor_index:segment.anchor_index + 1
                ]
                if float(candidate.sum().item()) > 0.0:
                    seed = candidate
            if seed is None and conditioning is None:
                raise ValueError("anchor 确认遮罩为空且没有文本提示，无法追踪")
            segment_threshold = detection_threshold
            if seed is not None and seeded_spawn_threshold > 0.0:
                segment_threshold = seeded_spawn_threshold
            track_data = SAM3_VideoTrack.execute(
                images=frames,
                model=model,
                initial_mask=seed,
                conditioning=conditioning,
                detection_threshold=segment_threshold,
                max_objects=max_objects,
                detect_interval=detect_interval,
            ).args[0]
            object_indices = ""
            if (
                seed is not None
                and seed_identity_max_distance > 0.0
                and track_data.get("packed_masks") is not None
            ):
                object_indices = _seed_identity_object_indices(
                    frames,
                    track_data,
                    seed_count=int(seed.shape[0]),
                    max_distance=float(seed_identity_max_distance),
                )
            segment_masks = SAM3_TrackToMask.execute(
                track_data=track_data,
                object_indices=object_indices,
            ).args[0].to(dtype=torch.float32, device="cpu")
            if segment.reverse:
                segment_masks = torch.flip(segment_masks, dims=[0])
            output[segment.emit_start:segment.emit_stop] = segment_masks[
                segment.emit_start - segment.start:segment.emit_stop - segment.start
            ]
        if override_anchor_frames and resized_anchor_masks is not None:
            for anchor_position, frame_index in enumerate(anchor_frames):
                confirmed = resized_anchor_masks[anchor_position]
                if float(confirmed.sum().item()) > 0.0:
                    output[frame_index] = confirmed
        return io.NodeOutput(output)


class SAM3DemoExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SAM3BoundingBoxes,
            SAM3MultiPromptDetectCached,
            SAM3AnchoredKeyframeDetectCached,
            SAM3InjectConfirmedKeyMasks,
            SAM3ValidateMask,
            SAM3CleanMask,
            SAM3ClipMaskToBox,
            SAM3MaskBatch,
            SAM3ComposeEditMask,
            SAM3TrackEditMask,
            SAM3SelectKeyframes,
            SAM3TrackVideoMasks,
            SAM3OpticalFlowMasks,
            SAM3GateKeyMasks,
            SAM3UnionMasks,
            SAM3TemporalSmoothMasks,
            SAM3MaskTrackBoxes,
            SAM3CropImagesToBox,
            SAM3PasteMaskToCanvas,
            SAM3RefineMaskWithPoints,
            SAM3GreenScreenImage,
            SAM3GreenScreenVideo,
        ]


async def comfy_entrypoint() -> SAM3DemoExtension:
    return SAM3DemoExtension()
