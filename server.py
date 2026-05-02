from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
from pathlib import Path

from fastmcp import FastMCP

logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config & model loader  (must be above the startup hook)
# ---------------------------------------------------------------------------

WEIGHTS_DIR  = Path(__file__).parent / "weights"
WEIGHTS_FILE = WEIGHTS_DIR / "model.safetensors"

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

mcp = FastMCP("fundus-cup-disc")

@mcp.on_startup()
async def load_model_on_startup():
    logger.info("Pre-loading SegFormer at startup...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _get_model)
    logger.info("Model ready.")
