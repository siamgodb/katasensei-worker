"""Worker settings, all from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    katago_binary: str
    katago_model: str
    katago_config: str
    katago_human_model: str | None
    laravel_url: str
    callback_secret: str
    api_token: str
    default_visits: int
    analysis_timeout_seconds: float

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            katago_binary=os.environ.get("KATAGO_BINARY", "katago"),
            katago_model=os.environ["KATAGO_MODEL"],
            katago_config=os.environ["KATAGO_CONFIG"],
            # Unset means no rank comparison at all, for any tier — which is a
            # valid way to run a cheap box, not a misconfiguration.
            katago_human_model=os.environ.get("KATAGO_HUMAN_MODEL") or None,
            laravel_url=os.environ.get("LARAVEL_URL", "http://127.0.0.1:8000"),
            callback_secret=os.environ["CALLBACK_SECRET"],
            api_token=os.environ["API_TOKEN"],
            default_visits=int(os.environ.get("DEFAULT_VISITS", "100")),
            # How long the analysis board may hold the connection. Long enough
            # for a deep query on a slow CPU box, short enough that a wedged
            # engine does not tie up a queue worker indefinitely.
            analysis_timeout_seconds=float(os.environ.get("ANALYSIS_TIMEOUT", "60")),
        )
