"""ProxyServer — expose a pymmcore-plus core over HTTP + WebSocket.

Routes:
    POST /rpc              — call any method on the core
    POST /mda/exec_event   — execute a single MDA event (configure + snap)
    GET  /health           — liveness check
    WS   /signals          — stream core signals to connected clients
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import numpy as np
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from ._serialize import decode, encode

logger = logging.getLogger("pymmcore_proxy.server")

# Signals to forward from core.events
_CORE_SIGNALS = [
    "propertyChanged",
    "configSet",
    "exposureChanged",
    "stagePositionChanged",
    "XYStagePositionChanged",
    "systemConfigurationLoaded",
]

# Signals to forward from core.mda.events
_MDA_SIGNALS = [
    "frameReady",
    "sequenceStarted",
    "sequenceFinished",
    "sequenceCanceled",
    "sequencePauseToggled",
]


class ProxyServer:
    """Wraps a CMMCorePlus/UniMMCore and serves it over HTTP + WebSocket."""

    def __init__(self, core, host: str = "127.0.0.1", port: int = 5600):
        self.core = core
        self.host = host
        self.port = port
        self._ws_clients: set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._start_time = time.time()
        self._command_count = 0
        self._recent_commands: list[dict] = []

        self.app = Starlette(
            routes=[
                Route("/rpc", self._handle_rpc, methods=["POST"]),
                Route("/mda/exec_event", self._handle_mda_event, methods=["POST"]),
                Route("/health", self._handle_health, methods=["GET"]),
                Route("/stats", self._handle_stats, methods=["GET"]),
                WebSocketRoute("/signals", self._handle_signals),
            ],
            on_startup=[self._on_startup],
        )

    def _on_startup(self):
        self._loop = asyncio.get_event_loop()
        self._connect_signal_forwarding()

    def _connect_signal_forwarding(self):
        """Connect to core signals and forward them over WebSocket."""
        events = getattr(self.core, "events", None)
        if events:
            for name in _CORE_SIGNALS:
                sig = getattr(events, name, None)
                if sig is not None:
                    sig.connect(self._make_signal_handler("events", name))

        mda = getattr(self.core, "mda", None)
        if mda:
            mda_events = getattr(mda, "events", None)
            if mda_events:
                for name in _MDA_SIGNALS:
                    sig = getattr(mda_events, name, None)
                    if sig is not None:
                        sig.connect(self._make_signal_handler("mda.events", name))

    def _make_signal_handler(self, group: str, name: str):
        def handler(*args):
            self._broadcast_signal(group, name, args)
        return handler

    def _broadcast_signal(self, group: str, name: str, args: tuple):
        if not self._ws_clients or self._loop is None:
            return
        try:
            msg = json.dumps({
                "group": group,
                "signal": name,
                "args": [encode(a) for a in args],
            })
        except Exception as e:
            logger.warning(f"Failed to serialize signal {group}.{name}: {e}")
            return
        for ws in list(self._ws_clients):
            try:
                asyncio.run_coroutine_threadsafe(ws.send_text(msg), self._loop)
            except Exception:
                self._ws_clients.discard(ws)

    # ------------------------------------------------------------------
    # HTTP handlers
    # ------------------------------------------------------------------

    async def _handle_health(self, request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def _handle_stats(self, request: Request) -> JSONResponse:
        return JSONResponse({
            "uptime_s": time.time() - self._start_time,
            "command_count": self._command_count,
            "recent_commands": self._recent_commands[-20:],
        })

    def _record_command(self, method: str):
        self._command_count += 1
        self._recent_commands.append({"t": time.time(), "method": method})
        if len(self._recent_commands) > 50:
            self._recent_commands = self._recent_commands[-50:]

    def _resolve_dotted(self, path: str) -> Any:
        """Resolve a dotted path like 'mda.engine.setup_event' on the core."""
        obj = self.core
        for part in path.split("."):
            obj = getattr(obj, part)
        return obj

    async def _handle_rpc(self, request: Request) -> JSONResponse:
        body = await request.json()
        action = body.get("action", "call")

        try:
            if action == "getattr":
                attr = body.get("attr", "")
                self._record_command(f"getattr:{attr}")
                result = self._resolve_dotted(attr)
                return JSONResponse({"ok": True, "value": encode(result)})

            if action == "setattr":
                attr = body.get("attr", "")
                value = decode(body.get("value"))
                self._record_command(f"setattr:{attr}")
                # Split into parent path + final attr name
                parts = attr.rsplit(".", 1)
                if len(parts) == 2:
                    parent = self._resolve_dotted(parts[0])
                    setattr(parent, parts[1], value)
                else:
                    setattr(self.core, attr, value)
                return JSONResponse({"ok": True, "value": None})

            # Default: method call
            method = body.get("method", "")
            args = [decode(a) for a in body.get("args", [])]
            kwargs = {k: decode(v) for k, v in body.get("kwargs", {}).items()}
            self._record_command(method)

            func = self._resolve_dotted(method)
            result = await asyncio.to_thread(func, *args, **kwargs)
            return JSONResponse({"ok": True, "value": encode(result)})
        except Exception as e:
            return JSONResponse(
                {"ok": False, "error": str(e), "error_type": type(e).__name__},
                status_code=200,  # keep 200 — error is in the payload
            )

    async def _handle_mda_event(self, request: Request) -> JSONResponse:
        """Execute a single MDA event: configure hardware + snap."""
        body = await request.json()
        event_data = body.get("event", {})
        self._record_command("mda.exec_event")

        try:
            result = self._execute_event(event_data)
            return JSONResponse({"ok": True, "value": encode(result)})
        except Exception as e:
            return JSONResponse(
                {"ok": False, "error": str(e), "error_type": type(e).__name__},
            )

    def _execute_event(self, event_data: dict) -> dict[str, Any]:
        """Configure hardware per MDAEvent spec, snap, return image + metadata."""
        core = self.core

        # Channel
        channel = event_data.get("channel")
        if channel and isinstance(channel, dict):
            group = channel.get("group", "")
            config = channel.get("config", "")
            if group and config:
                core.setConfig(group, config)

        # Exposure
        exposure = event_data.get("exposure")
        if exposure is not None:
            core.setExposure(float(exposure))

        # XY position
        x = event_data.get("x_pos")
        y = event_data.get("y_pos")
        if x is not None and y is not None:
            core.setXYPosition(float(x), float(y))
            core.waitForDevice(core.getXYStageDevice())

        # Z position
        z = event_data.get("z_pos")
        if z is not None:
            try:
                focus_dev = core.getFocusDevice()
                core.setPosition(focus_dev, float(z))
                core.waitForDevice(focus_dev)
            except Exception:
                pass

        # Snap
        core.snapImage()
        image = core.getImage()

        return {
            "image": image,
            "metadata": {
                "time_utc": time.time(),
                "index": event_data.get("index", {}),
            },
        }

    # ------------------------------------------------------------------
    # WebSocket handler
    # ------------------------------------------------------------------

    async def _handle_signals(self, websocket: WebSocket):
        await websocket.accept()
        self._ws_clients.add(websocket)
        try:
            while True:
                # Keep alive — client doesn't send messages (for now)
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            self._ws_clients.discard(websocket)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self):
        """Start the server (blocking)."""
        import uvicorn
        uvicorn.run(self.app, host=self.host, port=self.port)


def serve(core, host: str = "127.0.0.1", port: int = 5600):
    """Convenience: create and run a ProxyServer."""
    server = ProxyServer(core, host=host, port=port)
    server.run()
