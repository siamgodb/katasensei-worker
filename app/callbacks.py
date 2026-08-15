"""Sending results back to Laravel.

Laravel does not poll the worker: results are pushed as they are produced, so
a player watching a progress bar sees it move while the review is running
rather than all at once when it ends.

On RunPod this crosses the public internet. There is no private network to sit
behind any more and no fixed address to allow, so the signature is the whole
of the lock rather than the second one — which is why it now covers a timestamp
as well as the body. Without that, a batch captured once can be replayed later,
and the final batch of a review is the message that marks it finished.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from typing import Any

import httpx

log = logging.getLogger(__name__)


def sign(secret: str, timestamp: str, body: bytes) -> str:
    """An HMAC over the exact bytes that will be sent, and when.

    Over the body rather than over a summary of it: signing anything less lets
    the parts that were not signed be changed in transit. The timestamp is
    inside the signature rather than beside it, or moving it would be all an
    attacker needed to make an old batch look current.
    """
    return hmac.new(
        secret.encode(),
        timestamp.encode() + b"." + body,
        hashlib.sha256,
    ).hexdigest()


PROBE = b"katasensei-ping"
"""A fixed, public string. Only the key used to sign it is secret."""


def proof(secret: str) -> str:
    """Something both ends can compute and neither has to send the secret for.

    The last silent way a review can be lost. The worker reaches Laravel, runs
    the game, produces every report — and the callback is refused because the
    two copies of the signing key do not match. Nothing about that is visible
    until the reports fail to arrive, by which time the card has been paid for.

    An HMAC over a constant, truncated: equal on both sides means the keys are
    equal, and the value carries nothing an attacker cannot already compute
    from any real callback they observe.
    """
    return hmac.new(secret.encode(), PROBE, hashlib.sha256).hexdigest()[:16]


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

        timestamp = str(int(time.time()))

        response = await self._client.post(
            f"{self._base_url}/api/internal/reviews/{review_id}/reports",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Timestamp": timestamp,
                "X-Signature": sign(self._secret, timestamp, body),
            },
        )

        response.raise_for_status()

        log.info(
            "delivered %d report(s) for review %s (%s)",
            len(reports),
            review_id,
            progress.get("status"),
        )
