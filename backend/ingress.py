"""Bound HTTP bodies before application parsing or execution."""

from __future__ import annotations

import asyncio
import math

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_TOO_LARGE = {"error": {"code": "request_too_large", "message": "Request body too large"}}
_BAD_LENGTH = {"error": {"code": "invalid_content_length", "message": "Invalid Content-Length"}}
_BUSY = {"error": {
    "code": "ingress_capacity_unavailable", "message": "Request ingress is busy"
}}
_TIMED_OUT = {"error": {"code": "request_timeout", "message": "Request body timed out"}}
MAX_BODY_READERS = 64
MIN_BODY_TIMEOUT_S = 0.1
MAX_BODY_TIMEOUT_S = 30


class _IngressBusy(Exception):
    pass


class BodyLimitMiddleware:
    """Read at most the configured budget before allowing downstream execution.

    Pre-reading is necessary even for routes that never consume their body.
    No reader tasks are spawned; cancellation belongs to the ASGI request task.
    """

    def __init__(
        self, app: ASGIApp, max_body_bytes: int, *,
        max_body_readers: int = 8, body_timeout_s: float = 5,
    ) -> None:
        if (
            type(max_body_readers) is not int
            or not 1 <= max_body_readers <= MAX_BODY_READERS
            or not math.isfinite(body_timeout_s)
            or not MIN_BODY_TIMEOUT_S <= body_timeout_s <= MAX_BODY_TIMEOUT_S
        ):
            raise ValueError("Invalid body-reader settings")
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.max_body_readers = max_body_readers
        self.body_timeout_s = body_timeout_s
        self.active_readers = 0

    def _declared_length(self, scope: Scope) -> int | None:
        lengths = [value for name, value in scope["headers"] if name.lower() == b"content-length"]
        if not lengths:
            return None
        if len(lengths) != 1 or any(
            name.lower() == b"transfer-encoding" for name, _ in scope["headers"]
        ):
            raise ValueError
        digits = lengths[0].strip(b" \t")
        if not digits.isdigit():
            raise ValueError
        digits = digits.lstrip(b"0") or b"0"
        # Avoid arbitrary-size integer conversion on attacker-controlled headers.
        if len(digits) > len(str(self.max_body_bytes)):
            raise OverflowError
        length = int(digits)
        if length > self.max_body_bytes:
            raise OverflowError
        return length

    async def _read_body(self, receive: Receive, deadline: float) -> bytes | None:
        body = bytearray()
        loop = asyncio.get_running_loop()
        while True:
            # Even an immediately-ready receive returning endless empty frames
            # must yield to cancellation/other requests and obey the total bound.
            await asyncio.sleep(0)
            if loop.time() >= deadline:
                raise TimeoutError
            message = await receive()
            if asyncio.current_task().cancelling():
                # A receive implementation that swallowed cancellation must
                # not turn a cancelled upload into downstream execution.
                raise asyncio.CancelledError
            if loop.time() >= deadline:
                raise TimeoutError
            if message.get("type") == "http.disconnect":
                return None
            if message.get("type") != "http.request":
                raise ValueError
            chunk = message.get("body", b"")
            if not isinstance(chunk, bytes) or type(message.get("more_body", False)) is not bool:
                raise ValueError
            if len(body) + len(chunk) > self.max_body_bytes:
                raise OverflowError
            body.extend(chunk)
            if not message.get("more_body", False):
                return bytes(body)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        try:
            declared = self._declared_length(scope)
            # No await between checking and reserving: atomic on one event loop.
            if self.active_readers >= self.max_body_readers:
                raise _IngressBusy
            self.active_readers += 1
            try:
                deadline = asyncio.get_running_loop().time() + self.body_timeout_s
                async with asyncio.timeout_at(deadline):
                    body = await self._read_body(receive, deadline)
                    if body is None:  # disconnected before the complete body arrived
                        return
                    if declared is not None and declared != len(body):
                        raise ValueError
            finally:
                # The original receive has unwound. Downstream replay, inference
                # and SSE do not own upload capacity or its timeout context.
                self.active_readers -= 1
        except _IngressBusy:
            await JSONResponse(_BUSY, status_code=503)(scope, receive, send)
            return
        except TimeoutError:
            await JSONResponse(_TIMED_OUT, status_code=408)(scope, receive, send)
            return
        except OverflowError:
            await JSONResponse(_TOO_LARGE, status_code=413)(scope, receive, send)
            return
        except ValueError:
            await JSONResponse(_BAD_LENGTH, status_code=400)(scope, receive, send)
            return

        async def replay() -> Message:
            nonlocal body
            if body is not None:
                message = {"type": "http.request", "body": body, "more_body": False}
                body = None
                return message
            # Preserve the original disconnect channel for streaming responses.
            return await receive()

        await self.app(scope, replay, send)
