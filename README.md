# ComfyUI-SAM3-GreenScreen

Production-tested ComfyUI custom nodes for **video product / subject green-screen masking**, built on top of ComfyUI's native SAM 3.1 nodes. Designed for e-commerce short videos: track a confirmed subject (product, person, garment) across hard cuts, hand-held/worn state changes and close-ups, then composite a green screen.

面向电商短视频的「商品/主体绿幕遮罩」ComfyUI 自定义节点包：基于 ComfyUI 原生 SAM 3.1 tracker，支持锚点种子、跨镜头追踪、时序防闪烁与绿幕合成。

## Requirements

| Dependency | Version |
|---|---|
| ComfyUI | >= 0.33 (must include native SAM3 nodes: `comfy_extras/nodes_sam3.py`) |
| Checkpoint | `sam3.1_multiplex_fp16.safetensors` in `models/checkpoints/` ([Comfy-Org/sam3.1](https://huggingface.co/Comfy-Org/sam3.1)) |
| Python deps | `av`, `numpy`, `pillow` (usually already present in a ComfyUI env) |

> SAM 3 / 3.1 model weights are distributed by Meta under the SAM License — this repo contains **code only**.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/zriyox/ComfyUI-SAM3-GreenScreen sam3_demo_web
# restart ComfyUI
```

## Core node: `SAM3_TrackVideoMasks`

Full-video mask tracking that wraps ComfyUI core's `SAM3_VideoTrack` / `SAM3_TrackToMask` with anchor semantics:

- **Anchor seeding** — feed confirmed masks (`anchor_masks`, one per anchor) with their frame indices (`anchor_frames_json`); each anchor seeds its own tracking segment, and frames before the first anchor are tracked in reverse (equivalent bidirectional propagation).
- **Per-frame concept re-detection** — pass text `conditioning` so the detector re-acquires the subject after hard cuts (new instances auto-spawn).
- **`seeded_spawn_threshold`** — when a seed guarantees recall of the confirmed instance, raise the spawn threshold for text-detected extra instances (default suggestion `0.6`) to suppress look-alike false positives (e.g. a same-category product in a different color) without hurting multi-instance products.
- **`override_anchor_frames`** — anchor frames output exactly the confirmed masks.

Inputs: `images`, `model` (from `CheckpointLoaderSimple` on the SAM 3.1 checkpoint), optional `conditioning` / `anchor_masks`, `anchor_frames_json`, `detection_threshold` (0.35 works well for small products), `seeded_spawn_threshold`, `max_objects` (0 = internal cap 64 — small caps get exhausted by stale cross-cut tracks), `detect_interval`.

Output: `MASK` batch `[N,H,W]` (union of tracked objects per frame).

## Supporting nodes

| Node | Purpose |
|---|---|
| `SAM3_TemporalSmoothMasks` | Per-shot temporal median (±radius frames, never across cuts) — kills single-frame flicker on thin/small targets |
| `SAM3_CleanMask` | Component filtering, hole filling, edge feathering |
| `SAM3_RefineMaskWithPoints` | Positive/negative point refinement on a base mask |
| `SAM3_GreenScreenVideo` / `SAM3_GreenScreenImage` | Green-screen composite + video encode |
| `SAM3_UnionMasks` / `SAM3_MaskBatch` / `SAM3_ComposeEditMask` / `InvertMask`-friendly outputs | Multi-branch mask combination (multi-part foregrounds, background inversion) |
| `SAM3_ValidateMask` / `SAM3_ClipMaskToBox` / `SAM3_BoundingBoxes` | Guardrails and box utilities |

Legacy keyframe/optical-flow nodes (`SAM3_AnchoredKeyframeDetectCached`, `SAM3_OpticalFlowMasks`, `SAM3_GateKeyMasks`, `SAM3_SelectKeyframes`) are kept for compatibility but superseded by `SAM3_TrackVideoMasks`.

## Minimal video workflow (API format)

```jsonc
{
  "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sam3.1_multiplex_fp16.safetensors"}},
  "2": {"class_type": "LoadVideo",            "inputs": {"file": "input.mp4"}},
  "3": {"class_type": "GetVideoComponents",   "inputs": {"video": ["2", 0]}},
  "4": {"class_type": "CLIPTextEncode",       "inputs": {"clip": ["1", 1], "text": "gold earring"}},
  "5": {"class_type": "SAM3_TrackVideoMasks", "inputs": {
         "images": ["3", 0], "model": ["1", 0], "conditioning": ["4", 0],
         "anchor_frames_json": "[]", "detection_threshold": 0.35,
         "seeded_spawn_threshold": 0.0, "max_objects": 0,
         "detect_interval": 1, "override_anchor_frames": true}},
  "6": {"class_type": "SAM3_TemporalSmoothMasks", "inputs": {
         "images": ["3", 0], "masks": ["5", 0], "radius": 1,
         "interval": 120, "forced_frames_json": "[]", "scene_threshold": 0.12}},
  "7": {"class_type": "SAM3_GreenScreenVideo", "inputs": {
         "images": ["3", 0], "masks": ["6", 0], "fps": ["3", 2],
         "green_hex": "#00B140", "crf": 18}}
}
```

Multi-concept prompts use ComfyUI's comma syntax (`"lipstick, lipstick tube"`, max 32 tokens per concept). Exclusions (e.g. hands) are handled by running a second text-only `SAM3_TrackVideoMasks` and subtracting via `MaskComposite`.

## Field notes / tuning

- Small products (earrings, rings, lip gloss — 0.1%–3% of the frame): keep `detection_threshold` at 0.35 and the full 1008px input; the per-frame detector re-acquires after every hard cut, no shot splitting needed.
- `max_objects` small values (e.g. 4) get exhausted by stale cross-cut tracks and silently kill later shots — leave at 0.
- VRAM is constant w.r.t. video length (streaming tracker); a 16 GB GPU handles 720p full videos comfortably (~650 frames ≈ 95 s on an RTX 5070 Ti).

## License

MIT (code). SAM 3 / 3.1 weights are licensed separately by Meta.
