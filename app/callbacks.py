"""Sending results back to Laravel.

The worker sits on a private network and Laravel does not poll it: results are
pushed as they are produced. Each request is signed so the receiving endpoint
can tell a real batch from anything else that reaches it, without the app and
the worker sharing a session.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from typing import Any

import httpx

log = logging.getLogger(__name__)


def sign(secret: str, body: bytes) -> str:
    """An HMAC over the exact bytes that will be sent.

    Over the body rather than over a summary of it: signing anything less lets
    the parts that were not signed be changed in transit.
    """
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


class LaravelCallback:
    def __init__(self, base_url: str, secret: str, *, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._secret = secret
        self._client = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()

    async def send_batch(
        self,
        review_id: str,
        reports: list[dict[str, Any]],
        progress: dict[str, Any],
    ) -> None:
        body = json.dumps(
            {"reports": reports, "progress": progress},
            separators=(",", ":"),
        ).encode()

        response = await self._client.post(
            f"{self._base_url}/api/internal/reviews/{review_id}/reports",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Signature": sign(self._secret, body),
            },
        )

        response.raise_for_status()

        log.info(
            "delivered %d report(s) for review %s (%s)",
            len(reports),
            review_id,
            progress.get("status"),
        )
