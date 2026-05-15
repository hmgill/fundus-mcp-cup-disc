"""
server.py — fundus-mcp-cup-disc
================================
FastMCP server exposing SegFormer optic cup/disc segmentation as an MCP tool,
deployed via Prefect Horizon.

Image validation runs locally; GPU inference is dispatched to a Modal
serverless endpoint so Horizon doesn't need a GPU or model weights.

Required environment variables:
    MODAL_ENDPOINT_URL  Full Modal endpoint base URL,
                        e.g. https://<workspace>--cup-disc-segmentation-fastapi-app.modal.run

Optional environment variables:
    FASTMCP_DOCKET_URL  rediss://<host>:<port>  Redis for background tasks

Tools:
    segment_cup_disc(image_b64, image_id) → mask stats + base64 NPZ
    health()                              → liveness check
"""

from __future__ import annotations

import io
import json
import logging
import os

from fastmcp import FastMCP

logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Modal client — single blocking POST, no polling needed
# ---------------------------------------------------------------------------

def _modal_dispatch(image_id: str, image_b64: str) -> dict:
    import requests

    endpoint_url = os.environ.get("MODAL_ENDPOINT_URL", "").rstrip("/")
    if not endpoint_url:
        raise RuntimeError("MODAL_ENDPOINT_URL is not set.")

    logger.info(f"[{image_id}] Dispatching to Modal: {endpoint_url}/segment")

    resp = requests.post(
        f"{endpoint_url}/segment",
        json={"image_id": image_id, "image_b64": image_b64},
        timeout=120,
    )
    resp.raise_for_status()
    output = resp.json()

    if not output.get("success"):
        raise RuntimeError(f"Modal inference failed: {output.get('error')}")

    logger.info(
        f"[{image_id}] CDR={output.get('cdr')}  "
        f"payload≈{len(output.get('masks_b64', '')) / 1024:.1f}KB"
    )
    return output


# ---------------------------------------------------------------------------
# Image validation — runs locally, no GPU needed
# ---------------------------------------------------------------------------

def _validate_image(image_b64: str, image_id: str) -> tuple[int, int]:
    import base64
    from PIL import Image as _Image

    try:
        img_bytes = base64.b64decode(image_b64)
        img = _Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return img.size
    except Exception as e:
        raise RuntimeError(f"[{image_id}] Invalid image: {e}") from e


# ---------------------------------------------------------------------------
# FastMCP app
# ---------------------------------------------------------------------------

mcp = FastMCP("fundus-cup-disc")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def segment_cup_disc(image_b64: str, image_id: str) -> str:
    """
    Run SegFormer optic cup and disc segmentation on a fundus image.

    Image validation runs locally; GPU inference is dispatched to Modal.

    Args:
        image_b64:  Base64-encoded RGB fundus image (JPEG or PNG).
        image_id:   Identifier for this image.

    Returns:
        JSON with disc/cup pixel counts, CDR, and base64-encoded NPZ
        containing disc_annulus, cup, full_disc, and cd_raw arrays.

    Label map (from Modal worker):
        0 = background
        1 = disc annulus (outer ring, excludes cup)
        2 = optic cup
    """
    try:
        try:
            w, h = _validate_image(image_b64, image_id)
        except RuntimeError as e:
            return json.dumps({"success": False, "reason": str(e)})

        logger.info(f"[{image_id}] Image validated: {w}x{h}")
        output = _modal_dispatch(image_id, image_b64)
        return json.dumps(output)

    except Exception as e:
        logger.error(f"segment_cup_disc failed: {e}", exc_info=True)
        return json.dumps({"success": False, "error": str(e), "image_id": image_id})


@mcp.tool()
async def health() -> str:
    """Liveness probe. Reports Modal endpoint configuration status."""
    endpoint_url = os.environ.get("MODAL_ENDPOINT_URL", "")
    return json.dumps({
        "status":  "ok",
        "service": "fundus-cup-disc",
        "modal": {
            "endpoint_url": endpoint_url or "(not set)",
            "configured":   bool(endpoint_url),
        },
    })


if __name__ == "__main__":
    mcp.run(stateless_http=True, json_response=True)
