"""Request-scoped Laya routing for Hermes gateway turns.

The router is advisory only: it never changes profiles, sessions, tool
permissions, or authorizes actions. Any failure fails upward to FULL.
"""
from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any, Callable

logger = logging.getLogger(__name__)

LAYA_ROUTE_URL = "http://127.0.0.1:6768/route"
VALID_LANES = frozenset({"FAST", "FULL", "HEAVY"})
DEFAULT_LANE = "FULL"


def route_hermes_request(
    text: str,
    *,
    url: str = LAYA_ROUTE_URL,
    timeout: float = 0.75,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> str:
    """Return FAST/FULL/HEAVY; malformed or unavailable routing returns FULL."""
    try:
        body = json.dumps({"text": text, "target": "hermes"}).encode("utf-8")
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        with opener(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                return DEFAULT_LANE
            payload = json.loads(response.read().decode("utf-8"))
        lane = payload.get("hermes", {}).get("class")
        if lane is None:
            lane = payload.get("class")
        lane = str(lane or "").upper()
        return lane if lane in VALID_LANES else DEFAULT_LANE
    except Exception as exc:
        logger.debug("Laya Hermes routing failed; using FULL: %s", exc)
        return DEFAULT_LANE


def reasoning_for_lane(lane: str, current: dict | None) -> dict | None:
    """Map a lane to a supported per-turn reasoning config."""
    if lane == "FAST":
        from hermes_constants import parse_reasoning_effort
        return parse_reasoning_effort("low")
    if lane == "HEAVY":
        from hermes_constants import parse_reasoning_effort
        return parse_reasoning_effort("high")
    return current
