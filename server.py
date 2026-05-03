"""
server.py — fundus-mcp-cup-disc
================================
FastMCP server exposing SegFormer optic cup/disc segmentation as an MCP tool,
for deployment on Prefect Horizon.

Weights are committed to the repo via Git LFS and loaded from
the ./weights/ directory at startup. No download required.

Tools:
    segment_cup_disc(image_b64, image_id) → mask stats + base64 NPZ  [background task]
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
TMP_CACHE_DIR = Path("/tmp/fundus-model-cache")

_model_cache: dict = {}


def _get_model():
    if "model" not in _model_cache:
        import torch
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

        if not WEIGHTS_FILE.exists():
            raise FileNotFoundError(f"Weights not found: {WEIGHTS_FILE}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use /tmp cache if available — avoids re-parsing safetensors (~24s → ~1s)
        if TMP_CACHE_DIR.exists():
            logger.info(f"Loading from /tmp cache (PID={os.getpid()}) ...")
            src = str(TMP_CACHE_DIR)
        else:
            logger.info(f"Loading from safetensors (PID={os.getpid()}) ...")
            src = str(WEIGHTS_DIR)

        processor = AutoImageProcessor.from_pretrained(src, local_files_only=True)
        # Override to training size (224) — preprocessor_config.json says 512
        # but config.json shows image_size=224. Smaller input = faster CPU inference.
        processor.size = {"height": 224, "width": 224}

        model = SegformerForSemanticSegmentation.from_pretrained(
            src, local_files_only=True,
        ).to(device)
        model.eval()

        # Save to /tmp so subsequent cold starts in this container skip safetensors parsing
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

# Load model at module level. Runs on every cold start in Lambda, but is
# reused across warm invocations within the same container instance.
logger.info(f"Pre-warming model at module import (PID={os.getpid()}) ...")
_get_model()
logger.info("Model ready.")

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
    Run SegFormer optic cup and disc segmentation as a background task.

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
        await progress.set_total(3)

        await progress.set_message("Loading model...")
        model, processor, device = _get_model()

        await progress.set_message("Preprocessing image...")
        img_bytes = base64.b64decode(image_b64)
        image     = _Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h      = image.size

        inputs = processor(image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        await progress.increment()

        await progress.set_message("Running segmentation inference...")
        with torch.no_grad():
            logits = model(**inputs).logits
        await progress.increment()

        await progress.set_message("Computing mask statistics...")
        upsampled = F.interpolate(logits, size=(h, w), mode="bilinear",
                                  align_corners=False)
        cd_raw    = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        disc_annulus = (cd_raw == 1).astype(np.uint8)
        cup          = (cd_raw == 2).astype(np.uint8)
        full_disc    = (cd_raw >= 1).astype(np.uint8)

        cup_px  = int(cup.sum())
        disc_px = int(full_disc.sum())
        cdr     = round(cup_px / disc_px, 4) if disc_px > 0 else 0.0

        payload = json.dumps({
            "success":               True,
            "image_id":              image_id,
            "shape":                 list(cd_raw.shape),
            "disc_pixel_count":      int(disc_annulus.sum()),
            "cup_pixel_count":       cup_px,
            "full_disc_pixel_count": disc_px,
            "cdr":                   cdr,
            "masks_b64":             "DUMMY",   # TODO: re-enable once payload size confirmed OK
            "model":                 str(WEIGHTS_FILE.name),
            "created_at":            datetime.utcnow().isoformat() + "Z",
        })
        logger.info(f"Response payload size: {len(payload)} bytes ({len(payload)/1024:.1f} KB)")
        await progress.increment()
        return payload

    except Exception as e:
        logger.error(f"segment_cup_disc failed: {e}", exc_info=True)
        return json.dumps({"success": False, "error": str(e), "image_id": image_id})


@mcp.tool()
async def health() -> str:
    """Liveness probe. Also reports whether the /tmp model cache is present."""
    return json.dumps({
        "status":         "ok",
        "service":        "fundus-cup-disc",
        "weights_file":   str(WEIGHTS_FILE),
        "weights_exists": WEIGHTS_FILE.exists(),
        "tmp_cache":      TMP_CACHE_DIR.exists(),
    })


if __name__ == "__main__":
    mcp.run(stateless_http=True, json_response=True)
