"""
server.py — fundus-mcp-cup-disc
================================
FastMCP server exposing SegFormer optic cup/disc segmentation as an MCP tool.

The tool returns a synchronous JSON-RPC response — no background task polling.
This is required for compatibility with the Agents SDK's MCPServerStreamableHttp,
which expects standard JSON-RPC responses.

Weights are committed to the repo via Git LFS and loaded from
the ./weights/ directory at startup. No download required.

Tools:
    segment_cup_disc(image_b64, image_id) -> mask stats + base64 NPZ
    health()                              -> liveness check
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from pathlib import Path

from fastmcp import FastMCP

logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config & model loader
# ---------------------------------------------------------------------------

WEIGHTS_DIR   = Path(__file__).parent / "weights"
WEIGHTS_FILE  = WEIGHTS_DIR / "model.safetensors"
TMP_CACHE_DIR = Path("/tmp/fundus-model-cache")

_model_cache: dict = {}


def _get_model():
    if "model" not in _model_cache:
        import torch
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

        if not WEIGHTS_FILE.exists():
            raise FileNotFoundError(f"Weights not found: {WEIGHTS_FILE}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if TMP_CACHE_DIR.exists():
            logger.info(f"Loading from /tmp cache (PID={os.getpid()}) ...")
            src = str(TMP_CACHE_DIR)
        else:
            logger.info(f"Loading from safetensors (PID={os.getpid()}) ...")
            src = str(WEIGHTS_DIR)

        processor = AutoImageProcessor.from_pretrained(src, local_files_only=True)
        processor.size = {"height": 224, "width": 224}

        model = SegformerForSemanticSegmentation.from_pretrained(
            src, local_files_only=True,
        ).to(device)
        model.eval()

        if not TMP_CACHE_DIR.exists():
            logger.info(f"Saving parsed weights to {TMP_CACHE_DIR} ...")
            model.save_pretrained(str(TMP_CACHE_DIR))
            processor.save_pretrained(str(TMP_CACHE_DIR))
            logger.info("Cache saved.")

        _model_cache["model"]     = model
        _model_cache["processor"] = processor
        _model_cache["device"]    = device
        logger.info(f"SegFormer ready on {device}.")

    return _model_cache["model"], _model_cache["processor"], _model_cache["device"]


# ---------------------------------------------------------------------------
# FastMCP app
# ---------------------------------------------------------------------------

logger.info(f"Pre-warming model at module import (PID={os.getpid()}) ...")
_get_model()
logger.info("Model ready.")

mcp = FastMCP("fundus-cup-disc")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def segment_cup_disc(image_b64: str, image_id: str) -> str:
    """
    Run SegFormer optic cup and disc segmentation.

    Args:
        image_b64:  Base64-encoded RGB fundus image (JPEG or PNG).
        image_id:   Identifier for this image.

    Returns:
        JSON with disc/cup pixel counts, CDR, and base64-encoded NPZ
        containing disc_annulus, cup, full_disc, and cd_raw arrays.

    Label map:
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
        model, processor, device = _get_model()

        img_bytes = base64.b64decode(image_b64)
        image     = _Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h      = image.size

        inputs = processor(image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        upsampled = F.interpolate(logits, size=(h, w), mode="bilinear",
                                  align_corners=False)
        cd_raw    = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        disc_annulus = (cd_raw == 1).astype(np.uint8)
        cup          = (cd_raw == 2).astype(np.uint8)
        full_disc    = (cd_raw >= 1).astype(np.uint8)

        cup_px  = int(cup.sum())
        disc_px = int(full_disc.sum())
        cdr     = round(cup_px / disc_px, 4) if disc_px > 0 else 0.0

        npz_buf = io.BytesIO()
        np.savez_compressed(
            npz_buf,
            disc_annulus=disc_annulus,
            cup=cup,
            full_disc=full_disc,
            cd_raw=cd_raw,
        )
        masks_b64 = base64.b64encode(npz_buf.getvalue()).decode()

        payload = json.dumps({
            "success":               True,
            "image_id":              image_id,
            "shape":                 list(cd_raw.shape),
            "disc_pixel_count":      int(disc_annulus.sum()),
            "cup_pixel_count":       cup_px,
            "full_disc_pixel_count": disc_px,
            "cdr":                   cdr,
            "masks_b64":             masks_b64,
            "model":                 str(WEIGHTS_FILE.name),
            "created_at":            datetime.utcnow().isoformat() + "Z",
        })
        logger.info(f"segment_cup_disc: {image_id}  CDR={cdr}  payload={len(payload)/1024:.1f}KB")
        return payload

    except Exception as e:
        logger.error(f"segment_cup_disc failed: {e}", exc_info=True)
        return json.dumps({"success": False, "error": str(e), "image_id": image_id})


@mcp.tool()
async def health() -> str:
    """Liveness probe. Reports whether model weights and /tmp cache are present."""
    return json.dumps({
        "status":         "ok",
        "service":        "fundus-cup-disc",
        "weights_file":   str(WEIGHTS_FILE),
        "weights_exists": WEIGHTS_FILE.exists(),
        "tmp_cache":      TMP_CACHE_DIR.exists(),
    })


if __name__ == "__main__":
    mcp.run(stateless_http=True, json_response=True)
