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
from dataclasses import dataclass
from typing import Any

from .analysis import Analyst, PositionRequest
from .callbacks import LaravelCallback
from .config import Settings
from .engine import KataGoEngine
from .query import ReviewRequest
from .reviewer import Reviewer
from .schemas import AnalyzeIn, ReviewIn

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


async def handler(job: dict[str, Any]) -> dict[str, Any]:
    payload = dict(job.get("input") or {})
    kind = str(payload.pop("kind", "")).lower()

    if kind not in {"review", "analyze"}:
        # Raised rather than returned: RunPod records a raised handler as a
        # failed job, which is what the reconciler on the Laravel side reads to
        # decide a review is never coming.
        raise ValueError(f"unknown kind {kind!r}, expected 'review' or 'analyze'")

    current = await runtime()

    if kind == "analyze":
        return await analyse(current, AnalyzeIn.model_validate(payload))

    return await review(current, ReviewIn.model_validate(payload))


async def analyse(current: Runtime, body: AnalyzeIn) -> dict[str, Any]:
    return await current.analyst.analyse(
        PositionRequest(
            query_id=body.query_id,
            moves=[(m.color, m.loc) for m in body.moves],
            board_x_size=body.board_x_size,
            board_y_size=body.board_y_size,
            komi=body.komi,
            rules=body.rules,
            max_visits=body.max_visits,
            initial_stones=[(s.color, s.loc) for s in body.initial_stones] or None,
            include_ownership=body.include_ownership,
        )
    )


async def review(current: Runtime, body: ReviewIn) -> dict[str, Any]:
    job = await current.reviewer.run(
        ReviewRequest(
            review_id=body.review_id,
            moves=[(m.color, m.loc) for m in body.moves],
            board_x_size=body.board_x_size,
            board_y_size=body.board_y_size,
            komi=body.komi,
            rules=body.rules,
            initial_stones=[(s.color, s.loc) for s in body.initial_stones] or None,
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
