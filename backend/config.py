"""Application configuration, evaluated once when the server imports main."""

from __future__ import annotations

import math
import os

from backend.admission import AdmissionBudget
from backend.providers import AnthropicProvider, MockProvider, OpenAIProvider, Provider

DEFAULT_MAX_BODY_BYTES = 262144
MAX_CONFIGURED_BODY_BYTES = 1048576


def build_body_limit() -> int | None:
    """Bound public HTTP parsing; developer mode preserves existing behavior."""
    mode = os.environ.get("AI_SYSTEMS_LAB_MODE", "developer")
    if mode == "developer":
        return None
    if mode != "public-demo":
        raise ValueError("AI_SYSTEMS_LAB_MODE must be 'developer' or 'public-demo'")
    try:
        raw = os.environ.get("AI_SYSTEMS_LAB_MAX_BODY_BYTES", str(DEFAULT_MAX_BODY_BYTES))
        if not raw.isascii() or not raw.isdecimal():
            raise ValueError
        value = int(raw)
        if not 1 <= value <= MAX_CONFIGURED_BODY_BYTES:
            raise ValueError
        return value
    except ValueError:
        raise ValueError(
            "AI_SYSTEMS_LAB_MAX_BODY_BYTES must be an integer between 1 and 1048576"
        ) from None


def build_providers() -> list[Provider]:
    """Construct only mock providers for public-demo; preserve developer defaults."""
    mode = os.environ.get("AI_SYSTEMS_LAB_MODE", "developer")
    if mode == "public-demo":
        return [MockProvider()]
    if mode == "developer":
        return [AnthropicProvider(), OpenAIProvider(), MockProvider()]
    raise ValueError("AI_SYSTEMS_LAB_MODE must be 'developer' or 'public-demo'")


def build_admission() -> AdmissionBudget | None:
    """Only the public app receives bounded work and HTTP-waiter budgets."""
    mode = os.environ.get("AI_SYSTEMS_LAB_MODE", "developer")
    if mode == "developer":
        return None
    if mode != "public-demo":
        raise ValueError("AI_SYSTEMS_LAB_MODE must be 'developer' or 'public-demo'")

    def limit(name: str, default: int, minimum: int) -> int:
        raw = os.environ.get(name, str(default))
        if not raw.isascii() or not raw.isdecimal() or int(raw) < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
        return int(raw)

    def duration(name: str, default: float) -> float:
        try:
            value = float(os.environ.get(name, str(default)))
            if not math.isfinite(value) or value <= 0:
                raise ValueError
            return value
        except ValueError:
            raise ValueError(f"{name} must be a finite positive duration in seconds") from None

    return AdmissionBudget(
        max_active=limit("AI_SYSTEMS_LAB_MAX_ACTIVE", 4, 1),
        max_pending=limit("AI_SYSTEMS_LAB_MAX_PENDING", 16, 0),
        max_requests=limit("AI_SYSTEMS_LAB_MAX_REQUESTS", 64, 1),
        work_timeout_s=duration("AI_SYSTEMS_LAB_WORK_TIMEOUT_S", 10),
        stream_timeout_s=duration("AI_SYSTEMS_LAB_STREAM_TIMEOUT_S", 15),
        shutdown_grace_s=duration("AI_SYSTEMS_LAB_SHUTDOWN_GRACE_S", 5),
        cancel_grace_s=duration("AI_SYSTEMS_LAB_CANCEL_GRACE_S", 1),
    )
