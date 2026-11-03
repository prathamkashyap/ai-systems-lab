"""Application provider selection, evaluated once when the server imports main."""

from __future__ import annotations

import os

from backend.admission import AdmissionBudget
from backend.providers import AnthropicProvider, MockProvider, OpenAIProvider, Provider


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

    return AdmissionBudget(
        max_active=limit("AI_SYSTEMS_LAB_MAX_ACTIVE", 4, 1),
        max_pending=limit("AI_SYSTEMS_LAB_MAX_PENDING", 16, 0),
        max_requests=limit("AI_SYSTEMS_LAB_MAX_REQUESTS", 64, 1),
    )
