"""
server.py — fundus-mcp-cup-disc
================================
FastMCP server exposing SegFormer optic cup/disc segmentation as an MCP tool,
deployed via Prefect Horizon.

Preprocessing (base64 decode + PIL validation) runs locally; GPU inference is
dispatched to a RunPod serverless endpoint so Horizon doesn't need a GPU or
model weights.

Required environment variables:
    RUNPOD_API_KEY       RunPod API key
    RUNPOD_ENDPOINT_URL  Full RunPod endpoint base URL,
                         e.g. https://api.runpod.ai/v2/<endpoint_id>

Optional environment variables:
    FASTMCP_DOCKET_URL   rediss://<host>:<port>  Redis for background tasks
    RUNPOD_POLL_INTERVAL Seconds between status polls (default: 3)
    RUNPOD_MAX_WAIT      Seconds before timeout (default: 120)

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
import time
from datetime import timedelta

import requests
from fastmcp import FastMCP
from fastmcp.dependencies import Progress
from fastmcp.server.tasks import TaskConfig

logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RUNPOD_API_KEY       = os.environ.get("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_URL  = os.environ.get("RUNPOD_ENDPOINT_URL", "").rstrip("/")
RUNPOD_POLL_INTERVAL = int(os.environ.get("RUNPOD_POLL_INTERVAL", "3"))
RUNPOD_MAX_WAIT      = int(os.environ.get("RUNPOD_MAX_WAIT", "120"))

if not RUNPOD_API_KEY:
    logger.warning("RUNPOD_API_KEY is not set — inference calls will fail.")
if not RUNPOD_ENDPOINT_URL:
    logger.warning("RUNPOD_ENDPOINT_URL is not set — inference calls will fail.")

_redis_url = os.environ.get("FASTMCP_DOCKET_URL")
if not _redis_url:
    logger.warning(
        "FASTMCP_DOCKET_URL is not set — background tasks will fail across "
        "stateless HTTP requests. Set this to your Redis URL in Horizon env vars."
    )


# ---------------------------------------------------------------------------
# RunPod client
# ---------------------------------------------------------------------------

def _runpod_session() -> requests.Session:
    """Return a requests Session pre-configured with RunPod auth headers."""
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type":  "application/json",
    })
    return session


def _runpod_dispatch(image_id: str, image_b64: str) -> dict:
    """
    Submit a cup/disc segmentation job to the RunPod serverless endpoint and
    poll until complete.

    Args:
        image_id:   Identifier for logging and response correlation.
        image_b64:  Base64-encoded RGB fundus image (JPEG or PNG).

    Returns:
        The output dict from RunPod on success.
        Raises RuntimeError on job failure or TimeoutError on timeout.
    """
    session = _runpod_session()

    resp = session.post(
        f"{RUNPOD_ENDPOINT_URL}/run",
        json={"input": {
            "image_id":  image_id,
            "image_b64": image_b64,
        }},
    )
    resp.raise_for_status()
    job_id = resp.json().get("id")
    if not job_id:
        raise RuntimeError(f"No job ID in RunPod response: {resp.json()}")
    logger.info(f"[{image_id}] RunPod job submitted: {job_id}")

    deadline = time.time() + RUNPOD_MAX_WAIT
    while time.time() < deadline:
        resp = session.get(f"{RUNPOD_ENDPOINT_URL}/status/{job_id}")
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")
        logger.info(f"[{image_id}] RunPod status: {status}")
        if status == "COMPLETED":
            output = data.get("output", {})
            if not output.get("success"):
                raise RuntimeError(
                    f"RunPod job completed but reported failure: {output.get('error')}"
                )
            return output
        if status == "FAILED":
            raise RuntimeError(f"RunPod job failed: {data.get('error')}")
        time.sleep(RUNPOD_POLL_INTERVAL)

    raise TimeoutError(
        f"RunPod job {job_id} did not complete within {RUNPOD_MAX_WAIT}s"
    )


# ---------------------------------------------------------------------------
# Preprocessing — runs locally on Horizon, no GPU needed
# ---------------------------------------------------------------------------

def _validate_image(image_b64: str, image_id: str) -> tuple[int, int]:
    """
    Decode and open the image to confirm it's a valid RGB fundus image.
    Returns (width, height). Raises RuntimeError if invalid.

    This is intentionally lightweight — pixel preprocessing (resize, normalise)
    happens on the RunPod worker alongside the model, keeping the two steps
    co-located and avoiding a double encode/decode round-trip.
    """
    from PIL import Image as _Image

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

@mcp.tool(task=TaskConfig(mode="required", poll_interval=timedelta(seconds=5)))
async def segment_cup_disc(
    image_b64: str,
    image_id: str,
    progress: Progress = Progress(),
) -> str:
    """
    Run SegFormer optic cup and disc segmentation on a fundus image.

    Image validation runs locally; GPU inference is dispatched to RunPod.

    Args:
        image_b64:  Base64-encoded RGB fundus image (JPEG or PNG).
        image_id:   Identifier for this image.

    Returns:
        JSON with disc/cup pixel counts, CDR, and base64-encoded NPZ
        containing disc_annulus, cup, full_disc, and cd_raw arrays.

    Label map (from RunPod worker):
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

        await progress.set_message("Running cup/disc segmentation on RunPod...")
        output = _runpod_dispatch(image_id, image_b64)
        await progress.increment()

        await progress.set_message("Done.")
        logger.info(
            f"[{image_id}] CDR={output.get('cdr')}  "
            f"payload≈{len(output.get('masks_b64', '')) / 1024:.1f}KB"
        )
        await progress.increment()

        # Pass the RunPod output straight through — it already contains the
        # full payload (pixel counts, CDR, masks_b64, shape, created_at).
        return json.dumps(output)

    except Exception as e:
        logger.error(f"segment_cup_disc failed: {e}", exc_info=True)
        return json.dumps({"success": False, "error": str(e), "image_id": image_id})


@mcp.tool()
async def health() -> str:
    """Liveness probe. Reports RunPod endpoint configuration status."""
    return json.dumps({
        "status":  "ok",
        "service": "fundus-cup-disc",
        "runpod": {
            "endpoint_url":    RUNPOD_ENDPOINT_URL or "(not set)",
            "api_key_present": bool(RUNPOD_API_KEY),
            "configured":      bool(RUNPOD_API_KEY and RUNPOD_ENDPOINT_URL),
        },
    })


if __name__ == "__main__":
    mcp.run(stateless_http=True, json_response=True)
