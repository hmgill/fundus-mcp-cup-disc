"""
server.py — fundus-mcp-cup-disc
================================
FastMCP server exposing SegFormer optic cup/disc segmentation as an MCP tool,
for deployment on Prefect Horizon (stateless HTTP, multi-process).

Architecture reality on Prefect Horizon / stateless Lambda:
  - Multiple worker processes start in parallel at cold start.
  - Each process independently imports this module and runs module-level code.
  - Stateless HTTP means each request MAY land in a different process.
  - _model_cache dicts are NOT shared across processes.

Strategy:
  1. Module-level load: each worker loads the model into its own _model_cache.
     With multiple workers, this is concurrent but each worker only loads once.
  2. /tmp torch checkpoint cache: after the first full safetensors parse (~20s),
     the state_dict is saved to /tmp/segformer_cached.pt. Subsequent workers
     load from /tmp (~2-3s) instead of re-parsing safetensors (~20s).
  3. Dummy inference warm-up: forces cuDNN kernel selection before requests.
  4. Lifespan context: re-checks cache and loads if somehow missed at import.

Lambda constraints:
  - Synchronous response limit: 6 MB
  - NPZ masks off by default; set ENABLE_MASKS=1 env var to enable.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
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
TMP_CACHE_PT       = Path("/tmp/segformer_cached.pt")   # fast re-load path
ENABLE_MASKS       = os.environ.get("ENABLE_MASKS", "0") == "1"
MAX_RESPONSE_BYTES = 4 * 1024 * 1024   # 4 MB — headroom below Lambda 6 MB limit

_model_cache: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Model loader with /tmp checkpoint cache
# ---------------------------------------------------------------------------

def _load_model() -> None:
    """
    Load SegFormer into _model_cache.

    Load order (fastest first):
      1. Already in _model_cache — no-op.
      2. /tmp/segformer_cached.pt exists — load state_dict in ~2s.
      3. weights/model.safetensors — full parse ~20s, then save to /tmp.
    """
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Always load processor from config (tiny, fast)
    processor = AutoImageProcessor.from_pretrained(
        str(WEIGHTS_DIR), local_files_only=True,
    )

    if TMP_CACHE_PT.exists():
        # Fast path: load pre-parsed state_dict from /tmp (~2-3s)
        logger.info(f"Loading SegFormer state_dict from /tmp cache ({TMP_CACHE_PT}) ...")
        t0 = time.time()
        try:
            model = SegformerForSemanticSegmentation.from_pretrained(
                str(WEIGHTS_DIR), local_files_only=True,
                state_dict=torch.load(str(TMP_CACHE_PT), map_location=device),
            )
        except Exception as e:
            logger.warning(f"/tmp cache load failed ({e}), falling back to safetensors")
            TMP_CACHE_PT.unlink(missing_ok=True)
            return _load_model()   # recursive retry via slow path
        logger.info(f"Loaded from /tmp cache in {time.time()-t0:.1f}s")
    else:
        # Slow path: parse safetensors (~20s), then cache to /tmp
        logger.info(f"Loading SegFormer from safetensors ({WEIGHTS_FILE}) ...")
        t0 = time.time()
        model = SegformerForSemanticSegmentation.from_pretrained(
            str(WEIGHTS_DIR), local_files_only=True,
        )
        elapsed = time.time() - t0
        logger.info(f"Safetensors parsed in {elapsed:.1f}s. Saving to /tmp cache ...")
        try:
            torch.save(model.state_dict(), str(TMP_CACHE_PT))
            logger.info(f"Saved to {TMP_CACHE_PT} ({TMP_CACHE_PT.stat().st_size // 1024 // 1024} MB)")
        except Exception as e:
            logger.warning(f"Could not save /tmp cache: {e} (will re-parse next time)")

    model = model.to(device)
    model.eval()

    # Dummy inference warm-up — forces lazy PyTorch/cuDNN allocations
    logger.info("Dummy warm-up inference (64x64) ...")
    from PIL import Image as _Image
    dummy  = _Image.fromarray(np.zeros((64, 64, 3), dtype="uint8"))
    inputs = processor(dummy, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        _ = model(**inputs).logits

    _model_cache["model"]     = model
    _model_cache["processor"] = processor
    _model_cache["device"]    = device
    logger.info(f"SegFormer ready on {device}. PID={os.getpid()}")


# ---------------------------------------------------------------------------
# Module-level load — runs in each worker at import time
# ---------------------------------------------------------------------------

try:
    logger.info(f"Pre-warming model at module import (PID={os.getpid()}) ...")
    _load_model()
except Exception as _e:
    logger.error(f"Module-level model load failed: {_e}", exc_info=True)
    _model_cache["load_error"] = str(_e)


# ---------------------------------------------------------------------------
# FastMCP lifespan — safety net in case import-time load was skipped/failed
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastMCP):
    if "model" not in _model_cache:
        logger.info(f"Lifespan: model not in cache, loading now (PID={os.getpid()}) ...")
        try:
            _load_model()
        except Exception as e:
            logger.error(f"Lifespan model load failed: {e}", exc_info=True)
            _model_cache["load_error"] = str(e)
    else:
        logger.info(f"Lifespan: model already loaded (PID={os.getpid()}), skipping.")
    yield
    logger.info(f"Lifespan shutdown (PID={os.getpid()}).")


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
                    Resize to <=512x512 before encoding.
        image_id:   Identifier string echoed in the result JSON.

    Returns:
        JSON with disc/cup pixel counts, CDR, shape, and optionally NPZ masks.

    Label map:
        0 = background | 1 = disc annulus | 2 = optic cup
    """
    import torch
    import torch.nn.functional as F
    import numpy as np
    from PIL import Image as _Image
    from datetime import datetime

    try:
        # Attempt lazy load if somehow still missing (should not happen normally)
        if "model" not in _model_cache:
            logger.warning(f"Model missing at tool call time (PID={os.getpid()}), loading now...")
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

        await progress.set_total(3)

        # Step 1: decode & preprocess
        await progress.set_message("Preprocessing image...")
        img_bytes = base64.b64decode(image_b64)
        image     = _Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h      = image.size
        logger.info(f"[{image_id}] Input: {w}x{h} PID={os.getpid()}")

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

        # Optional NPZ (gated by env var + size check)
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
                logger.info(f"[{image_id}] NPZ included ({len(probe)//1024} KB)")
            else:
                logger.warning(f"[{image_id}] NPZ {len(probe)//1024} KB > budget, omitted")

        payload = json.dumps(result)
        logger.info(f"[{image_id}] Done {len(payload)//1024}KB CDR={cdr} cup={cup_px} disc={disc_px}")
        await progress.increment()
        return payload

    except Exception as e:
        logger.error(f"segment_cup_disc failed [{image_id}]: {e}", exc_info=True)
        return json.dumps({"success": False, "error": str(e), "image_id": image_id})


@mcp.tool()
async def health() -> str:
    """Liveness probe — reports model load status, device, cache paths."""
    import torch
    return json.dumps({
        "status":          "ok",
        "service":         "fundus-cup-disc",
        "pid":             os.getpid(),
        "weights_file":    str(WEIGHTS_FILE),
        "weights_exists":  WEIGHTS_FILE.exists(),
        "tmp_cache":       str(TMP_CACHE_PT),
        "tmp_cache_exists": TMP_CACHE_PT.exists(),
        "tmp_cache_mb":    round(TMP_CACHE_PT.stat().st_size / 1024 / 1024, 1) if TMP_CACHE_PT.exists() else 0,
        "model_loaded":    "model" in _model_cache,
        "load_error":      _model_cache.get("load_error"),
        "device":          str(_model_cache.get("device", "not loaded")),
        "masks_enabled":   ENABLE_MASKS,
        "cuda_available":  torch.cuda.is_available(),
    })


if __name__ == "__main__":
    mcp.run(stateless_http=True, json_response=True)
