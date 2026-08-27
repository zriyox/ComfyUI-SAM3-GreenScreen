"""Feature-based identity helpers for mask candidates."""

from __future__ import annotations

from typing import Any

import numpy as np


def _single_frame_feature_map(feature_map: Any):
    if getattr(feature_map, "ndim", None) == 3:
        return feature_map
    if getattr(feature_map, "ndim", None) == 4:
        if int(feature_map.shape[0]) != 1:
            raise ValueError("feature_map 必须是单帧特征")
        return feature_map[0]
    raise ValueError("feature_map 必须是 [C,H,W] 或 [1,C,H,W]")


def _single_frame_mask(mask: Any):
    if getattr(mask, "ndim", None) == 2:
        return mask
    if getattr(mask, "ndim", None) == 3:
        if int(mask.shape[0]) != 1:
            raise ValueError("mask 必须是单帧遮罩")
        return mask[0]
    raise ValueError("mask 必须是 [H,W] 或 [1,H,W]")


def _normalize_feature(feature):
    if hasattr(feature, "detach"):
        import torch

        norm = torch.linalg.vector_norm(feature)
        if float(norm.detach().cpu()) <= 1e-12:
            return None
        return feature / norm
    norm = float(np.linalg.norm(feature))
    if norm <= 1e-12:
        return None
    return feature / norm


def masked_feature_embedding(feature_map: Any, mask: Any):
    """Return an L2-normalized masked-average feature for one frame."""

    features = _single_frame_feature_map(feature_map)
    weights = _single_frame_mask(mask)
    if tuple(features.shape[-2:]) != tuple(weights.shape[-2:]):
        raise ValueError("feature_map 与 mask 的空间尺寸必须一致")

    if hasattr(features, "detach"):
        import torch

        weights = weights.to(device=features.device, dtype=features.dtype)
        weights = weights.clamp_min(0.0)
        total_weight = weights.sum()
        if float(total_weight.detach().cpu()) <= 1e-12:
            return None
        pooled = (features * weights.unsqueeze(0)).sum(dim=(-2, -1)) / total_weight
        return _normalize_feature(pooled)

    features = np.asarray(features, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    weights = np.maximum(weights, 0.0)
    total_weight = float(weights.sum())
    if total_weight <= 1e-12:
        return None
    pooled = (features * weights[None, ...]).sum(axis=(-2, -1)) / total_weight
    return _normalize_feature(pooled)


def masked_feature_cosine_similarity(
    reference_features: Any,
    reference_mask: Any,
    candidate_features: Any,
    candidate_mask: Any,
) -> float | None:
    """Compare two masked-average features without moving CUDA tensors to CPU."""

    reference = masked_feature_embedding(reference_features, reference_mask)
    candidate = masked_feature_embedding(candidate_features, candidate_mask)
    if reference is None or candidate is None:
        return None

    if hasattr(reference, "detach"):
        import torch

        return float(torch.dot(reference, candidate).detach().cpu())
    return float(np.dot(reference, candidate))


def masked_color_distance(
    reference_image: Any,
    reference_mask: Any,
    candidate_image: Any,
    candidate_mask: Any,
) -> float | None:
    """Return normalized RGB mean distance for two masked image regions."""

    reference = _masked_color_mean(reference_image, reference_mask)
    candidate = _masked_color_mean(candidate_image, candidate_mask)
    if reference is None or candidate is None:
        return None
    if hasattr(reference, "detach"):
        import torch

        return float(torch.linalg.vector_norm(reference - candidate).detach().cpu())
    return float(np.linalg.norm(reference - candidate))


def masked_color_signature_distance(
    reference_image: Any,
    reference_mask: Any,
    candidate_image: Any,
    candidate_mask: Any,
) -> float | None:
    """Compare masked mean RGB direction while ignoring global brightness."""

    reference = _masked_color_mean(reference_image, reference_mask)
    candidate = _masked_color_mean(candidate_image, candidate_mask)
    if reference is None or candidate is None:
        return None

    reference_signature = _normalize_feature(reference)
    candidate_signature = _normalize_feature(candidate)
    if reference_signature is None or candidate_signature is None:
        return None
    if hasattr(reference_signature, "detach"):
        import torch

        return float(
            torch.linalg.vector_norm(
                reference_signature - candidate_signature
            ).detach().cpu()
        )
    return float(np.linalg.norm(reference_signature - candidate_signature))


def _masked_color_mean(image: Any, mask: Any):
    if getattr(image, "ndim", None) == 4:
        if int(image.shape[0]) != 1:
            raise ValueError("image 必须是单帧图像")
        image = image[0]
    if getattr(image, "ndim", None) != 3 or int(image.shape[-1]) != 3:
        raise ValueError("image 必须是 [H,W,3] 或 [1,H,W,3]")
    mask = _single_frame_mask(mask)
    if tuple(image.shape[:2]) != tuple(mask.shape[-2:]):
        raise ValueError("image 与 mask 的空间尺寸必须一致")

    if hasattr(image, "detach"):
        import torch

        weights = mask.to(device=image.device, dtype=image.dtype).clamp_min(0.0)
        total_weight = weights.sum()
        if float(total_weight.detach().cpu()) <= 1e-12:
            return None
        return (image * weights.unsqueeze(-1)).sum(dim=(0, 1)) / total_weight

    image = np.asarray(image, dtype=np.float32)
    weights = np.maximum(np.asarray(mask, dtype=np.float32), 0.0)
    total_weight = float(weights.sum())
    if total_weight <= 1e-12:
        return None
    return (image * weights[..., None]).sum(axis=(0, 1)) / total_weight


def select_identity_candidate(
    candidates: list[dict],
    *,
    minimum_similarity: float,
    maximum_appearance_distance: float | None = None,
) -> dict | None:
    """Select the highest-identity candidate after applying a cosine Gate."""

    eligible = [
        candidate
        for candidate in candidates
        if float(candidate.get("similarity", -1.0)) >= float(minimum_similarity)
        and (
            maximum_appearance_distance is None
            or float(candidate.get("appearance_distance", float("inf")))
            <= float(maximum_appearance_distance)
        )
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda candidate: (
            float(candidate.get("similarity", -1.0)),
            float(candidate.get("score", -1.0)),
        ),
    )
