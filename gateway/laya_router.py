"""Request-scoped Laya routing for Hermes gateway turns.

The router is advisory only: it never changes profiles, sessions, tool
permissions, or authorizes actions. Any failure fails upward to FULL.
"""
from __future__ import annotations

import json
import json as jsonlib
import http.client
import logging
import threading
import time
import urllib.request
from urllib.parse import urlsplit
from typing import Any, Callable

logger = logging.getLogger(__name__)

LAYA_ROUTE_URL = "http://127.0.0.1:6768/route"
VALID_LANES = frozenset({"FAST", "FULL", "HEAVY"})
DEFAULT_LANE = "FULL"
FAST_MODEL = "lfm3:1.7b"


class _PersistentLocalClient:
    """Tiny serialized keep-alive client for the loopback Laya sidecar."""

    def __init__(self) -> None:
        self._connection: http.client.HTTPConnection | None = None
        self._lock = threading.Lock()

    def post(self, url: str, *, json: dict, timeout: float) -> Any:
        parsed = urlsplit(url)
        body = jsonlib.dumps(json).encode("utf-8")
        with self._lock:
            if self._connection is None:
                self._connection = http.client.HTTPConnection(
                    parsed.hostname or "127.0.0.1", parsed.port or 80, timeout=timeout
                )
            try:
                self._connection.request(
                    "POST", parsed.path, body=body,
                    headers={"Content-Type": "application/json"},
                )
                raw = self._connection.getresponse()
                payload = raw.read()
            except Exception:
                self._connection.close()
                self._connection = None
                raise

        class Response:
            status_code = raw.status

            @staticmethod
            def json() -> dict:
                return jsonlib.loads(payload.decode("utf-8"))

        return Response()


_HTTP_CLIENT = _PersistentLocalClient()


def route_hermes_request(
    text: str,
    *,
    url: str = LAYA_ROUTE_URL,
    timeout: float = 0.75,
    opener: Callable[..., Any] | None = None,
    client: Any | None = None,
) -> str:
    """Return FAST/FULL/HEAVY; malformed or unavailable routing returns FULL."""
    started = time.perf_counter()
    try:
        request_payload = {"text": text, "target": "hermes"}
        if opener is not None:
            body = json.dumps(request_payload).encode("utf-8")
            request = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"}, method="POST"
            )
            with opener(request, timeout=timeout) as response:
                status = getattr(response, "status", 200)
                if status != 200:
                    return DEFAULT_LANE
                payload = json.loads(response.read().decode("utf-8"))
        else:
            response = (client or _HTTP_CLIENT).post(url, json=request_payload, timeout=timeout)
            if response.status_code != 200:
                return DEFAULT_LANE
            payload = response.json()
        lane = payload.get("hermes", {}).get("class")
        if lane is None:
            lane = payload.get("class")
        lane = str(lane or "").upper()
        return lane if lane in VALID_LANES else DEFAULT_LANE
    except Exception as exc:
        logger.debug("Laya Hermes routing failed; using FULL: %s", exc)
        return DEFAULT_LANE
    finally:
        logger.info("Laya Hermes routing latency_ms=%.1f", (time.perf_counter() - started) * 1000)


def reasoning_for_lane(lane: str, current: dict | None) -> dict | None:
    """Map a lane to a supported per-turn reasoning config."""
    if lane == "FAST":
        from hermes_constants import parse_reasoning_effort
        return parse_reasoning_effort("low")
    if lane == "HEAVY":
        from hermes_constants import parse_reasoning_effort
        return parse_reasoning_effort("high")
    return current


def runtime_for_lane(
    lane: str,
    model: str,
    runtime: dict,
    *,
    resolver: Callable[..., dict] | None = None,
) -> tuple[str, dict]:
    """Use local LFM3 only for FAST; preserve the established route otherwise."""
    if lane != "FAST":
        return model, runtime
    try:
        if resolver is None:
            from gateway.run import _resolve_runtime_agent_kwargs_for_provider
            resolver = _resolve_runtime_agent_kwargs_for_provider
        local_runtime = resolver("ollama", target_model=FAST_MODEL)
        local_runtime = dict(local_runtime)
        local_runtime.pop("model", None)
        return FAST_MODEL, local_runtime
    except Exception as exc:
        logger.warning("LFM3 fast runtime unavailable; preserving configured runtime: %s", exc)
        return model, runtime
