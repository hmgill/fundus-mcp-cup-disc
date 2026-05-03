"""
server.py — fundus-mcp-cup-disc
================================
FastMCP server exposing SegFormer optic cup/disc segmentation as an MCP tool,
for deployment on Prefect Horizon (stateless HTTP, multi-process).

This version runs segment_cup_disc as a regular synchronous tool (no tasks/
background workers). The call blocks until inference completes and returns
the result directly. Simpler, fewer moving parts, no GetTaskRequest polling.

/tmp checkpoint cache:
  First worker parses safetensors (~20s) and saves to /tmp/segformer_cached.pt.
  Subsequent workers load from /tmp (~2-3s). Persists across worker spawns
  within the same Horizon deployment instance.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WEIGHTS_DIR        = Path(__file__).parent / "weights"
WEIGHTS_FILE       = WEIGHTS_DIR / "model.safetensors"
TMP_CACHE_PT       = Path("/tmp/segformer_cached.pt")
ENABLE_MASKS       = os.environ.get("ENABLE_MASKS", "0") == "1"
MAX_RESPONSE_BYTES = 4 * 1024 * 1024

_model_cache: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def _load_model() -> None:
    if "model" in _model_cache:
        return

    import torch
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
    import numpy as np

    if not WEIGHTS_FILE.exists():
        raise FileNotFoundError(
            f"Weights not found: {WEIGHTS_FILE}. "
            "Ensure weights/model.safetensors is committed via Git LFS."
        )

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(str(WEIGHTS_DIR), local_files_only=True)

    if TMP_CACHE_PT.exists():
        logger.info(f"Loading from /tmp cache (PID={os.getpid()}) ...")
        t0 = time.time()
        try:
            model = SegformerForSemanticSegmentation.from_pretrained(
                str(WEIGHTS_DIR), local_files_only=True,
                state_dict=torch.load(str(TMP_CACHE_PT), map_location=device),
            )
            logger.info(f"Loaded from /tmp in {time.time()-t0:.1f}s")
        except Exception as e:
            logger.warning(f"/tmp cache load failed ({e}), falling back to safetensors")
            TMP_CACHE_PT.unlink(missing_ok=True)
            return _load_model()
    else:
        logger.info(f"Loading from safetensors (PID={os.getpid()}) ...")
        t0 = time.time()
        model = SegformerForSemanticSegmentation.from_pretrained(
            str(WEIGHTS_DIR), local_files_only=True,
        )
        logger.info(f"Parsed safetensors in {time.time()-t0:.1f}s, saving to /tmp ...")
        try:
            torch.save(model.state_dict(), str(TMP_CACHE_PT))
            logger.info(f"Saved {TMP_CACHE_PT.stat().st_size // 1024 // 1024} MB to /tmp")
        except Exception as e:
            logger.warning(f"Could not save /tmp cache: {e}")

    model = model.to(device)
    model.eval()

    logger.info("Dummy warm-up inference ...")
    from PIL import Image as _Image
    dummy  = _Image.fromarray(__import__("numpy").zeros((64, 64, 3), dtype="uint8"))
    inputs = processor(dummy, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        _ = model(**inputs).logits

    _model_cache["model"]     = model
    _model_cache["processor"] = processor
    _model_cache["device"]    = device
    logger.info(f"SegFormer ready on {device} (PID={os.getpid()})")


# ---------------------------------------------------------------------------
# Module-level pre-warm
# ---------------------------------------------------------------------------

try:
    logger.info(f"Pre-warming model at module import (PID={os.getpid()}) ...")
    _load_model()
except Exception as _e:
    logger.error(f"Module-level model load failed: {_e}", exc_info=True)
    _model_cache["load_error"] = str(_e)


# ---------------------------------------------------------------------------
# Lifespan safety net
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastMCP):
    if "model" not in _model_cache:
        logger.info(f"Lifespan: loading model (PID={os.getpid()}) ...")
        try:
            _load_model()
        except Exception as e:
            logger.error(f"Lifespan load failed: {e}", exc_info=True)
            _model_cache["load_error"] = str(e)
    else:
        logger.info(f"Lifespan: model already loaded (PID={os.getpid()})")
    yield


mcp = FastMCP("fundus-cup-disc", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def segment_cup_disc(image_b64: str, image_id: str) -> str:
    """
    Run SegFormer optic cup and disc segmentation on a fundus image.

    Args:
        image_b64:  Base64-encoded RGB fundus image (JPEG or PNG).
                    Resize to <=512x512 before encoding.
        image_id:   Identifier string echoed in the result JSON.

    Returns:
        JSON with disc/cup pixel counts, CDR, and image shape.

    Label map:
        0 = background | 1 = disc annulus | 2 = optic cup
    """
    import torch
    import torch.nn.functional as F
    import numpy as np
    from PIL import Image as _Image
    from datetime import datetime

    try:
        if "model" not in _model_cache:
            logger.warning(f"Model missing at call time (PID={os.getpid()}), loading ...")
            _load_model()

        if "model" not in _model_cache:
            return json.dumps({
                "success":  False,
                "error":    f"Model not loaded: {_model_cache.get('load_error', 'unknown')}",
                "image_id": image_id,
            })

        model     = _model_cache["model"]
        processor = _model_cache["processor"]
        device    = _model_cache["device"]

        img_bytes = base64.b64decode(image_b64)
        image     = _Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h      = image.size
        logger.info(f"[{image_id}] Input: {w}x{h} PID={os.getpid()}")

        inputs = processor(image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        upsampled    = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
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
            "shape":                 list(cd_raw.shape),
            "disc_pixel_count":      int(disc_annulus.sum()),
            "cup_pixel_count":       cup_px,
            "full_disc_pixel_count": disc_px,
            "cdr":                   cdr,
            "model":                 str(WEIGHTS_FILE.name),
            "created_at":            datetime.utcnow().isoformat() + "Z",
            "masks_included":        False,
        }

        if ENABLE_MASKS:
            npz_buf = io.BytesIO()
            np.savez_compressed(npz_buf,
                disc_annulus=disc_annulus, cup=cup,
                full_disc=full_disc, cd_raw=cd_raw)
            npz_b64 = base64.b64encode(npz_buf.getvalue()).decode()
            probe   = json.dumps({**result, "masks_b64": npz_b64})
            if len(probe.encode()) <= MAX_RESPONSE_BYTES:
                result["masks_b64"]      = npz_b64
                result["masks_included"] = True
            else:
                logger.warning(f"[{image_id}] NPZ {len(probe)//1024} KB > budget, omitted")

        payload = json.dumps(result)
        logger.info(f"[{image_id}] Done {len(payload)//1024}KB CDR={cdr}")
        return payload

    except Exception as e:
        logger.error(f"segment_cup_disc failed [{image_id}]: {e}", exc_info=True)
        return json.dumps({"success": False, "error": str(e), "image_id": image_id})


@mcp.tool()
async def health() -> str:
    """Liveness probe — reports model load status, device, and cache info."""
    import torch
    return json.dumps({
        "status":           "ok",
        "service":          "fundus-cup-disc",
        "pid":              os.getpid(),
        "weights_file":     str(WEIGHTS_FILE),
        "weights_exists":   WEIGHTS_FILE.exists(),
        "tmp_cache":        str(TMP_CACHE_PT),
        "tmp_cache_exists": TMP_CACHE_PT.exists(),
        "tmp_cache_mb":     round(TMP_CACHE_PT.stat().st_size / 1024 / 1024, 1) if TMP_CACHE_PT.exists() else 0,
        "model_loaded":     "model" in _model_cache,
        "load_error":       _model_cache.get("load_error"),
        "device":           str(_model_cache.get("device", "not loaded")),
        "masks_enabled":    ENABLE_MASKS,
        "cuda_available":   torch.cuda.is_available(),
    })


if __name__ == "__main__":
    mcp.run(stateless_http=True, json_response=True)
