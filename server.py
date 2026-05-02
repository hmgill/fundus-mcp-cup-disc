"""
server.py — fundus-mcp-cup-disc
================================
FastMCP server exposing SegFormer optic cup/disc segmentation as an MCP tool,
for deployment on Prefect Horizon (AWS Lambda via lambda_http).

Key constraints:
  - Lambda synchronous response limit: 6 MB
  - Lambda streaming response limit: 20 MB
  - lambda_http panics on body write errors (OOM kills, timeout mid-write)
  - NPZ masks are DISABLED by default; enable via ENABLE_MASKS=1 env var

Weights are committed to the repo via Git LFS and loaded from
the ./weights/ directory at startup. No download required.

Tools:
    segment_cup_disc(image_b64, image_id) → mask stats + optional base64 NPZ  [background task]
    health()                              → liveness check
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from datetime import timedelta
from pathlib import Path

from fastmcp import FastMCP
from fastmcp.dependencies import Progress
from fastmcp.server.tasks import TaskConfig

logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config & model loader
# ---------------------------------------------------------------------------

WEIGHTS_DIR   = Path(__file__).parent / "weights"
WEIGHTS_FILE  = WEIGHTS_DIR / "model.safetensors"
ENABLE_MASKS  = os.environ.get("ENABLE_MASKS", "0") == "1"

# Lambda response size budget (leave headroom below the 6 MB sync limit)
MAX_RESPONSE_BYTES = 4 * 1024 * 1024   # 4 MB hard cap on serialized JSON

_model_cache: dict = {}


def _get_model():
    if "model" not in _model_cache:
        import torch
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

        if not WEIGHTS_FILE.exists():
            raise FileNotFoundError(f"Weights not found: {WEIGHTS_FILE}")
        logger.info(f"Loading model from {WEIGHTS_DIR} ...")
        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = AutoImageProcessor.from_pretrained(
            str(WEIGHTS_DIR), local_files_only=True,
        )
        model = SegformerForSemanticSegmentation.from_pretrained(
            str(WEIGHTS_DIR), local_files_only=True,
        ).to(device)
        model.eval()
        _model_cache["model"]     = model
        _model_cache["processor"] = processor
        _model_cache["device"]    = device
        logger.info(f"SegFormer ready on {device}.")

    return _model_cache["model"], _model_cache["processor"], _model_cache["device"]


# ---------------------------------------------------------------------------
# FastMCP app
# ---------------------------------------------------------------------------

# Pre-warm model at import time so cold-start penalty is paid once.
logger.info("Pre-warming model at module import...")
try:
    _get_model()
    logger.info("Model ready.")
except Exception as e:
    logger.error(f"Model pre-warm failed (will retry on first call): {e}")


mcp = FastMCP("fundus-cup-disc")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool(task=TaskConfig(mode="required", poll_interval=timedelta(seconds=5)))
async def segment_cup_disc(
    image_b64: str,
    image_id: str,
    progress: Progress = Progress(),
) -> str:
    """
    Run SegFormer optic cup and disc segmentation on a fundus image.

    Args:
        image_b64:  Base64-encoded RGB fundus image (JPEG or PNG).
                    Resize to ≤512×512 before encoding to stay within
                    Lambda's 6 MB response limit.
        image_id:   Identifier for this image (used in result JSON).

    Returns:
        JSON string with disc/cup pixel counts, CDR, image shape, and
        optionally a base64-encoded NPZ (only when ENABLE_MASKS=1 env var
        is set and the payload stays under 4 MB).

    Label map (cd_raw array):
        0 = background
        1 = disc annulus (outer ring, excludes cup)
        2 = optic cup
    """
    import torch
    import torch.nn.functional as F
    import numpy as np
    from PIL import Image as _Image
    from datetime import datetime

    try:
        await progress.set_total(3)

        # ── Step 1: load model ────────────────────────────────────────────
        await progress.set_message("Loading model...")
        model, processor, device = _get_model()
        await progress.increment()

        # ── Step 2: preprocess ────────────────────────────────────────────
        await progress.set_message("Preprocessing image...")
        img_bytes = base64.b64decode(image_b64)
        image     = _Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h      = image.size
        logger.info(f"[{image_id}] Input image size: {w}x{h}")

        inputs = processor(image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # ── Step 3: inference ─────────────────────────────────────────────
        await progress.set_message("Running segmentation inference...")
        with torch.no_grad():
            logits = model(**inputs).logits
        await progress.increment()

        # ── Step 4: post-process ──────────────────────────────────────────
        await progress.set_message("Computing mask statistics...")
        upsampled = F.interpolate(logits, size=(h, w), mode="bilinear",
                                  align_corners=False)
        cd_raw       = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        disc_annulus = (cd_raw == 1).astype(np.uint8)
        cup          = (cd_raw == 2).astype(np.uint8)
        full_disc    = (cd_raw >= 1).astype(np.uint8)

        cup_px  = int(cup.sum())
        disc_px = int(full_disc.sum())
        cdr     = round(cup_px / disc_px, 4) if disc_px > 0 else 0.0

        result: dict = {
            "success":               True,
            "image_id":              image_id,
            "shape":                 list(cd_raw.shape),   # [H, W]
            "disc_pixel_count":      int(disc_annulus.sum()),
            "cup_pixel_count":       cup_px,
            "full_disc_pixel_count": disc_px,
            "cdr":                   cdr,
            "model":                 str(WEIGHTS_FILE.name),
            "created_at":            datetime.utcnow().isoformat() + "Z",
            "masks_included":        False,
        }

        # Optionally attach NPZ — only if env var set AND payload fits in budget
        if ENABLE_MASKS:
            npz_buf = io.BytesIO()
            np.savez_compressed(
                npz_buf,
                disc_annulus=disc_annulus,
                cup=cup,
                full_disc=full_disc,
                cd_raw=cd_raw,
            )
            npz_b64 = base64.b64encode(npz_buf.getvalue()).decode()
            # Probe total payload size before committing
            probe = json.dumps({**result, "masks_b64": npz_b64})
            if len(probe.encode()) <= MAX_RESPONSE_BYTES:
                result["masks_b64"]     = npz_b64
                result["masks_included"] = True
                logger.info(f"[{image_id}] NPZ included ({len(probe)//1024} KB payload)")
            else:
                logger.warning(
                    f"[{image_id}] NPZ payload {len(probe)//1024} KB exceeds "
                    f"{MAX_RESPONSE_BYTES//1024} KB budget — masks omitted. "
                    "Reduce image size or increase MAX_RESPONSE_BYTES."
                )

        payload = json.dumps(result)
        logger.info(
            f"[{image_id}] Response: {len(payload)//1024} KB | "
            f"cup={cup_px}px disc={disc_px}px CDR={cdr}"
        )
        await progress.increment()
        return payload

    except Exception as e:
        logger.error(f"segment_cup_disc failed for {image_id!r}: {e}", exc_info=True)
        return json.dumps({"success": False, "error": str(e), "image_id": image_id})


@mcp.tool()
async def health() -> str:
    """Liveness probe — reports weights path and model load status."""
    import torch
    model_loaded = "model" in _model_cache
    return json.dumps({
        "status":         "ok",
        "service":        "fundus-cup-disc",
        "weights_file":   str(WEIGHTS_FILE),
        "weights_exists": WEIGHTS_FILE.exists(),
        "model_loaded":   model_loaded,
        "device":         str(_model_cache.get("device", "not loaded")),
        "masks_enabled":  ENABLE_MASKS,
        "cuda_available": torch.cuda.is_available(),
    })


if __name__ == "__main__":
    mcp.run(stateless_http=True, json_response=True)
