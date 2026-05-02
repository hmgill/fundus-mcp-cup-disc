"""
server.py — fundus-mcp-cup-disc
================================
FastMCP server exposing SegFormer optic cup/disc segmentation as an MCP tool,
for deployment on Prefect Horizon (AWS Lambda via lambda_http).

Key architecture notes:
  - Prefect Horizon forks worker processes AFTER module import, so module-level
    _get_model() calls populate a cache that is NOT visible in the worker.
  - Model loading must happen inside the FastMCP lifespan context, which runs
    inside the worker after fork, before any tool calls are served.
  - A dummy inference pass during lifespan forces all lazy PyTorch allocations
    (cuDNN benchmarks, kernel compilation, etc.) before the first real request.
  - Lambda timeout: if model load + dummy inference exceeds the function timeout,
    increase it in the Prefect Horizon deployment config (recommend >=60s).

Lambda constraints:
  - Synchronous response limit: 6 MB
  - NPZ masks disabled by default; set ENABLE_MASKS=1 env var to enable.
  - MAX_RESPONSE_BYTES: 4 MB hard cap (leaves headroom below 6 MB limit).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from fastmcp.dependencies import Progress
from fastmcp.server.tasks import TaskConfig

logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WEIGHTS_DIR        = Path(__file__).parent / "weights"
WEIGHTS_FILE       = WEIGHTS_DIR / "model.safetensors"
ENABLE_MASKS       = os.environ.get("ENABLE_MASKS", "0") == "1"
MAX_RESPONSE_BYTES = 4 * 1024 * 1024   # 4 MB — headroom below Lambda 6 MB limit

# Shared state populated during lifespan (inside the worker process after fork)
_STATE: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Model loader (called from lifespan, inside the worker)
# ---------------------------------------------------------------------------

def _load_model() -> None:
    """Load SegFormer into _STATE. Must be called from inside the worker process."""
    import torch
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
    import numpy as np

    if not WEIGHTS_FILE.exists():
        raise FileNotFoundError(
            f"Model weights not found: {WEIGHTS_FILE}\n"
            f"Ensure weights/model.safetensors is committed via Git LFS."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading SegFormer from {WEIGHTS_DIR} on {device} ...")

    processor = AutoImageProcessor.from_pretrained(
        str(WEIGHTS_DIR), local_files_only=True,
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        str(WEIGHTS_DIR), local_files_only=True,
    ).to(device)
    model.eval()

    _STATE["model"]     = model
    _STATE["processor"] = processor
    _STATE["device"]    = device
    logger.info(f"SegFormer loaded on {device}.")

    # Dummy inference: forces cuDNN kernel selection + lazy allocs before first request
    logger.info("Running dummy inference warm-up (64x64 black image)...")
    from PIL import Image as _Image
    dummy  = _Image.fromarray(np.zeros((64, 64, 3), dtype="uint8"))
    inputs = processor(dummy, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        _ = model(**inputs).logits
    logger.info("Warm-up complete. Worker ready.")


# ---------------------------------------------------------------------------
# FastMCP lifespan — runs inside the worker process after fork
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastMCP):
    """Load model during worker startup; clean up on shutdown."""
    logger.info("=== Worker lifespan startup: loading model ===")
    try:
        _load_model()
    except Exception as e:
        logger.error(f"Model load failed during lifespan: {e}", exc_info=True)
        _STATE["load_error"] = str(e)
    yield
    # Shutdown: free GPU memory
    if "model" in _STATE:
        try:
            import torch
            del _STATE["model"]
            del _STATE["processor"]
            if str(_STATE.get("device", "cpu")) != "cpu":
                torch.cuda.empty_cache()
        except Exception:
            pass
    logger.info("=== Worker lifespan shutdown complete ===")


# ---------------------------------------------------------------------------
# FastMCP app
# ---------------------------------------------------------------------------

mcp = FastMCP("fundus-cup-disc", lifespan=lifespan)


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
                    Resize to <=512x512 before encoding to keep payload small.
        image_id:   Identifier for this image (echoed in the result JSON).

    Returns:
        JSON string with disc/cup pixel counts, CDR, image shape, and
        optionally a base64-encoded NPZ (only when ENABLE_MASKS=1 env var
        is set and the payload stays under MAX_RESPONSE_BYTES).

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
        # Guard: model must have loaded during lifespan
        if "model" not in _STATE:
            load_error = _STATE.get("load_error", "unknown — check startup logs")
            return json.dumps({
                "success":  False,
                "error":    f"Model not loaded: {load_error}",
                "image_id": image_id,
            })

        model     = _STATE["model"]
        processor = _STATE["processor"]
        device    = _STATE["device"]

        await progress.set_total(3)

        # Step 1: decode & preprocess
        await progress.set_message("Preprocessing image...")
        img_bytes = base64.b64decode(image_b64)
        image     = _Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h      = image.size
        logger.info(f"[{image_id}] Input image: {w}x{h}")

        inputs = processor(image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        await progress.increment()

        # Step 2: inference
        await progress.set_message("Running segmentation inference...")
        with torch.no_grad():
            logits = model(**inputs).logits
        await progress.increment()

        # Step 3: post-process
        await progress.set_message("Computing mask statistics...")
        upsampled = F.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False
        )
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

        # Optional NPZ attachment
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
            probe   = json.dumps({**result, "masks_b64": npz_b64})
            if len(probe.encode()) <= MAX_RESPONSE_BYTES:
                result["masks_b64"]      = npz_b64
                result["masks_included"] = True
                logger.info(f"[{image_id}] NPZ included ({len(probe)//1024} KB)")
            else:
                logger.warning(
                    f"[{image_id}] NPZ payload {len(probe)//1024} KB exceeds "
                    f"{MAX_RESPONSE_BYTES//1024} KB budget — masks omitted."
                )

        payload = json.dumps(result)
        logger.info(
            f"[{image_id}] Done — {len(payload)//1024} KB | "
            f"cup={cup_px}px disc={disc_px}px CDR={cdr}"
        )
        await progress.increment()
        return payload

    except Exception as e:
        logger.error(f"segment_cup_disc failed [{image_id}]: {e}", exc_info=True)
        return json.dumps({"success": False, "error": str(e), "image_id": image_id})


@mcp.tool()
async def health() -> str:
    """
    Liveness probe.
    Returns model load status, device, weights path, and configuration.
    If model_loaded=false, check load_error for the failure reason.
    """
    import torch
    model_loaded = "model" in _STATE
    return json.dumps({
        "status":         "ok",
        "service":        "fundus-cup-disc",
        "weights_file":   str(WEIGHTS_FILE),
        "weights_exists": WEIGHTS_FILE.exists(),
        "model_loaded":   model_loaded,
        "load_error":     _STATE.get("load_error"),
        "device":         str(_STATE.get("device", "not loaded")),
        "masks_enabled":  ENABLE_MASKS,
        "cuda_available": torch.cuda.is_available(),
    })


if __name__ == "__main__":
    mcp.run(stateless_http=True, json_response=True)
