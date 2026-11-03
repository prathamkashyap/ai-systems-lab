"""Application provider selection, evaluated once when the server imports main."""

from __future__ import annotations

import os

from backend.providers import AnthropicProvider, MockProvider, OpenAIProvider, Provider


def build_providers() -> list[Provider]:
    """Construct only mock providers for public-demo; preserve developer defaults."""
    mode = os.environ.get("AI_SYSTEMS_LAB_MODE", "developer")
    if mode == "public-demo":
        return [MockProvider()]
    if mode == "developer":
        return [AnthropicProvider(), OpenAIProvider(), MockProvider()]
    raise ValueError("AI_SYSTEMS_LAB_MODE must be 'developer' or 'public-demo'")
