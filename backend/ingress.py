"""Bound HTTP bodies before application parsing or execution."""

from __future__ import annotations

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_TOO_LARGE = {"error": {"code": "request_too_large", "message": "Request body too large"}}
_BAD_LENGTH = {"error": {"code": "invalid_content_length", "message": "Invalid Content-Length"}}


class BodyLimitMiddleware:
    """Read at most the configured budget before allowing downstream execution.

    Pre-reading is necessary even for routes that never consume their body.
    No reader tasks are spawned; cancellation belongs to the ASGI request task.
    """

    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

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

    async def _read_body(self, receive: Receive) -> bytes | None:
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return None
            chunk = message.get("body", b"")
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
            body = await self._read_body(receive)
            if body is None:  # disconnected before the complete body arrived
                return
            if declared is not None and declared != len(body):
                raise ValueError
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
