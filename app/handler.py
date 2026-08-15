"""The RunPod serverless entry point.

RunPod's model is one job per worker: it hands the container a job, waits for
this function to return, and is free to stop the container the moment it does.
That shapes everything here.

**The engine outlives the job.** Loading a neural net onto a GPU is several
seconds, and paying that per position would make the analysis board unusable.
The runtime is built on the first job and kept for the container's life, so a
worker that is already awake answers in the time the search actually takes.
Scale-to-zero means somebody pays that load — the first request of a quiet
period — and the board says so rather than showing a spinner that looks
identical to a fast request.

**A review runs to completion inside the call.** Accepting one and returning
"queued" would hand the worker back to RunPod, which would then shut down the
engine mid-review. The platform's queue replaces the worker's own.

Results still reach Laravel over the signed callback rather than through this
function's return value: a job's output is capped well below a game's worth of
teaching reports, and the player is watching a progress bar that has to move
while the review is running, not after it.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .analysis import Analyst, PositionRequest
from .callbacks import LaravelCallback
from .config import Settings
from .engine import KataGoEngine
from .query import ReviewRequest
from .reviewer import Reviewer
from .schemas import AnalyzeIn, ReviewIn
from .version import VERSION, fingerprint

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@dataclass
class Runtime:
    settings: Settings
    engine: KataGoEngine
    reviewer: Reviewer
    analyst: Analyst
    callback: LaravelCallback

    @classmethod
    async def start(cls, settings: Settings) -> "Runtime":
        engine = KataGoEngine(
            binary=settings.katago_binary,
            model=settings.katago_model,
            config=settings.katago_config,
            human_model=settings.katago_human_model,
        )

        callback = LaravelCallback(settings.laravel_url, settings.callback_secret)

        await engine.start()

        # No reviewer.start(): the background queue task is for the long-lived
        # box. Here every review is run inline by the job that asked for it.
        return cls(
            settings=settings,
            engine=engine,
            reviewer=Reviewer(engine, callback.send_batch),
            analyst=Analyst(engine, timeout=settings.analysis_timeout_seconds),
            callback=callback,
        )

    async def close(self) -> None:
        await self.engine.stop()
        await self.callback.close()


_runtime: Runtime | None = None
_lock = asyncio.Lock()


async def runtime() -> Runtime:
    """The engine for this container, started once.

    Also the recovery path: an engine that died takes every later job on this
    worker down with it unless it is noticed and replaced, and a worker RunPod
    believes is healthy can be handed jobs for a long time.
    """
    global _runtime

    async with _lock:
        if _runtime is not None and not _runtime.engine.is_running:
            log.warning("the engine is gone; starting a new one")
            await _runtime.close()
            _runtime = None

        if _runtime is None:
            _runtime = await Runtime.start(Settings.from_env())

    return _runtime


KINDS = {"review", "analyze", "ping"}


async def handler(job: dict[str, Any]) -> dict[str, Any]:
    payload = dict(job.get("input") or {})
    kind = str(payload.pop("kind", "")).lower()

    if kind not in KINDS:
        # Raised rather than returned: RunPod records a raised handler as a
        # failed job, which is what the reconciler on the Laravel side reads to
        # decide a review is never coming.
        raise ValueError(f"unknown kind {kind!r}, expected one of {sorted(KINDS)}")

    # Before the engine, and without it. The point of a ping is to be answerable
    # by a worker whose configuration is wrong, since that is the case worth
    # asking about.
    if kind == "ping":
        return await ping(warm=bool(payload.get("warm")))

    current = await runtime()

    if kind == "analyze":
        return await analyse(current, AnalyzeIn.model_validate(payload))

    return await review(current, ReviewIn.model_validate(payload))


WATCHED_ENV = (
    "KATAGO_BINARY",
    "KATAGO_MODEL",
    "KATAGO_HUMAN_MODEL",
    "KATAGO_CONFIG",
    "LARAVEL_URL",
    "CALLBACK_SECRET",
    "DEFAULT_VISITS",
    "ANALYSIS_TIMEOUT",
    "KATAGO_LOG_STDERR",
)


async def ping(*, warm: bool = False) -> dict[str, Any]:
    """Everything worth knowing about this worker, for the price of a cold start.

    Every silent failure this endpoint has had was a configuration one: a build
    that did not land, a model path that was not there, a `LARAVEL_URL` still
    pointing at localhost so a finished review had nowhere to go. Each was found
    by running a real job and reading the wreckage, which is minutes of a rented
    card to learn something a container can report in a second.

    Set `warm` to load the neural net as well. That costs what a cold start
    costs and is worth it before a review — a card that cannot load the net
    should say so before a player is told their game is being looked at.
    """
    laravel_url = os.environ.get("LARAVEL_URL", "")

    report: dict[str, Any] = {
        "version": VERSION,
        # Compare with `python -m app.version` locally. If they differ, the
        # image predates the code, and nothing else in this report matters yet.
        "fingerprint": fingerprint(),
        "warm": _runtime is not None and _runtime.engine.is_running,
        # Names and whether they are set, never values: this comes back through
        # RunPod's API and into logs, and one of them is the callback secret.
        "env": {name: bool(os.environ.get(name)) for name in WATCHED_ENV},
        "files": {
            "binary": _file(os.environ.get("KATAGO_BINARY", "katago")),
            "model": _file(os.environ.get("KATAGO_MODEL")),
            "human_model": _file(os.environ.get("KATAGO_HUMAN_MODEL")),
            "config": _file(os.environ.get("KATAGO_CONFIG")),
        },
        "gpu": _gpu(),
        # A review delivers its results by calling Laravel back. If that call
        # cannot be made, the whole review is GPU time spent on an answer that
        # goes in the bin — and the worker only finds out at the end.
        "laravel": await _reachable(laravel_url),
    }

    if warm:
        try:
            current = await runtime()
            report["warm"] = current.engine.is_running
        except Exception as exc:  # noqa: BLE001 - the answer, not a failure
            report["warm"] = False
            report["engine_error"] = str(exc)

    return report


def _file(path: str | None) -> dict[str, Any]:
    if not path:
        return {"set": False}

    resolved = Path(path)

    # Size as well as existence: a model truncated by a build that ran out of
    # disk is a file that exists and an engine that will not start.
    if not resolved.exists():
        return {"set": True, "path": path, "exists": False}

    return {"set": True, "path": path, "exists": True, "bytes": resolved.stat().st_size}


def _gpu() -> dict[str, Any]:
    """Whether there is a card here at all, without importing a CUDA runtime."""
    driver = Path("/proc/driver/nvidia/version")

    if not driver.exists():
        return {"present": False}

    return {"present": True, "driver": driver.read_text().splitlines()[0].strip()}


async def _reachable(url: str) -> dict[str, Any]:
    if not url:
        return {"configured": False}

    # Localhost on a serverless worker is the worker. Named rather than left to
    # show up as a connection error, because it is the mistake that looks most
    # like a working configuration in the endpoint's settings page.
    local = any(host in url for host in ("127.0.0.1", "localhost", "0.0.0.0"))

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, follow_redirects=True)

        # Any answer at all is the thing being tested. A 404 from the right
        # server is a pass; the callback goes to a different path anyway.
        return {"configured": True, "url": url, "local": local, "status": response.status_code}
    except Exception as exc:  # noqa: BLE001 - reported rather than raised
        return {"configured": True, "url": url, "local": local, "error": str(exc)}


async def analyse(current: Runtime, body: AnalyzeIn) -> dict[str, Any]:
    return await current.analyst.analyse(
        PositionRequest(
            query_id=body.query_id,
            moves=[m.as_pair() for m in body.moves],
            board_x_size=body.board_x_size,
            board_y_size=body.board_y_size,
            komi=body.komi,
            rules=body.rules,
            max_visits=body.max_visits,
            initial_stones=[s.as_pair() for s in body.initial_stones] or None,
            include_ownership=body.include_ownership,
        )
    )


async def review(current: Runtime, body: ReviewIn) -> dict[str, Any]:
    job = await current.reviewer.run(
        ReviewRequest(
            review_id=body.review_id,
            moves=[m.as_pair() for m in body.moves],
            board_x_size=body.board_x_size,
            board_y_size=body.board_y_size,
            komi=body.komi,
            rules=body.rules,
            initial_stones=[s.as_pair() for s in body.initial_stones] or None,
            max_visits=body.max_visits or current.settings.default_visits,
            student_rank=body.student_rank,
            student_color=body.student_color,
            rank_gap=body.rank_gap,
            # Asking for the human model when this worker was not given one is
            # a configuration question, not a request failure: the review runs
            # and comes back marked as degraded.
            use_human_model=body.use_human_model
            and current.settings.katago_human_model is not None,
        )
    )

    return job.snapshot()


def main() -> None:  # pragma: no cover - the process entry point
    import runpod

    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":  # pragma: no cover
    main()
