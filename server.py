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
    segment_cup_disc(image_b64, image_id) → mask stats + base64 NPZ  [background task]
    health()                              → liveness check
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Only stdlib + fastmcp at the top level.
# `requests` and `pillow` are imported lazily inside functions so that
# `fastmcp inspect` (which runs in a minimal build env with only fastmcp
# installed) can parse the tool signatures without hitting ModuleNotFoundError.
# ---------------------------------------------------------------------------
import io
import json
import logging
import os
from datetime import timedelta

from fastmcp import FastMCP
from fastmcp.dependencies import Progress

logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Modal client — single blocking POST, no polling loop needed
# ---------------------------------------------------------------------------

def _modal_dispatch(image_id: str, image_b64: str) -> dict:
    """
    POST to the Modal /segment endpoint and block until inference completes.
    Modal handles its own internal queueing and GPU scheduling.

    Args:
        image_id:   Identifier for logging and response correlation.
        image_b64:  Base64-encoded RGB fundus image (JPEG or PNG).

    Returns:
        The output dict from Modal on success.
        Raises RuntimeError on HTTP error or reported inference failure.
        Raises TimeoutError if Modal does not respond within 120 seconds.
    """
    import requests  # lazy import — not available during fastmcp inspect

    endpoint_url = os.environ.get("MODAL_ENDPOINT_URL", "").rstrip("/")
    if not endpoint_url:
        raise RuntimeError("MODAL_ENDPOINT_URL is not set.")

    logger.info(f"[{image_id}] Dispatching to Modal: {endpoint_url}/segment")

    resp = requests.post(
        f"{endpoint_url}/segment",
        json={"image_id": image_id, "image_b64": image_b64},
        timeout=120,  # Modal handles queueing internally; one blocking call is enough
    )
    resp.raise_for_status()
    output = resp.json()

    if not output.get("success"):
        raise RuntimeError(
            f"Modal inference failed: {output.get('error')}"
        )

    logger.info(
        f"[{image_id}] CDR={output.get('cdr')}  "
        f"payload≈{len(output.get('masks_b64', '')) / 1024:.1f}KB"
    )
    return output


# ---------------------------------------------------------------------------
# Image validation — runs locally on Horizon, no GPU needed
# ---------------------------------------------------------------------------

def _validate_image(image_b64: str, image_id: str) -> tuple[int, int]:
    """
    Decode and open the image to confirm it is a valid RGB fundus image.
    Returns (width, height). Raises RuntimeError if invalid.
    """
    import base64
    from PIL import Image as _Image  # lazy import

    try:
        img_bytes = base64.b64decode(image_b64)
        img = _Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return img.size  # (w, h)
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
async def segment_cup_disc(
    image_b64: str,
    image_id: str,
    progress: Progress = Progress(),
) -> str:
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
        await progress.set_total(3)
        await progress.set_message("Validating image...")

        try:
            w, h = _validate_image(image_b64, image_id)
        except RuntimeError as e:
            return json.dumps({"success": False, "reason": str(e)})

        logger.info(f"[{image_id}] Image validated: {w}x{h}")
        await progress.increment()

        await progress.set_message("Running cup/disc segmentation on Modal...")
        output = _modal_dispatch(image_id, image_b64)
        await progress.increment()

        await progress.set_message("Done.")
        await progress.increment()

        # Pass the Modal output straight through — it already contains the
        # full payload (pixel counts, CDR, masks_b64, shape, created_at).
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
