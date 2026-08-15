"""Running a review from start to finish.

Reviews are queued one at a time. The engine already parallelises internally
across its analysis threads, and running two games at once on a CPU box does
not make either finish sooner — it makes both finish late, and makes the
progress bar lie about which.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .engine import KataGoEngine
from .protocol import QueryTracker, Response
from .query import ReviewRequest, build_query

log = logging.getLogger(__name__)

# Reports are sent back in batches. One request per move would be a few hundred
# round trips per game; waiting for the whole game would mean the player sees
# nothing for ten minutes and loses everything if the last call fails.
BATCH_SIZE = 20
BATCH_INTERVAL_SECONDS = 15.0

FINAL_DELIVERY_ATTEMPTS = 3
"""How hard to try on the batch that says the review is finished."""

RETRY_BACKOFF_SECONDS = 1.0
"""Doubled per attempt. A named constant so tests can set it to nothing."""

POLL_INTERVAL_SECONDS = 1.0
"""How often the review loop wakes to flush and to check the engine is alive."""

Sink = Callable[[str, list[dict[str, Any]], dict[str, Any]], Awaitable[None]]
"""Called with (review_id, reports, progress) to deliver a batch."""


@dataclass
class ReviewJob:
    request: ReviewRequest
    status: str = "queued"
    done: int = 0
    total: int = 0
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    reports: dict[int, dict[str, Any]] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        return {
            "review_id": self.request.review_id,
            "status": self.status,
            "done": self.done,
            "total": self.total,
            "error": self.error,
            "elapsed_seconds": round(
                (self.finished_at or time.monotonic()) - self.started_at, 1
            )
            if self.started_at
            else 0.0,
        }


class Reviewer:
    def __init__(self, engine: KataGoEngine, sink: Sink) -> None:
        self._engine = engine
        self._sink = sink
        self._queue: asyncio.Queue[ReviewJob] = asyncio.Queue()
        self._jobs: dict[str, ReviewJob] = {}
        self._worker: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run_forever(), name="reviewer")

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()

            with contextlib.suppress(asyncio.CancelledError):
                await self._worker

            self._worker = None

    def submit(self, request: ReviewRequest) -> ReviewJob:
        job = ReviewJob(request=request, total=len(request.moves))
        self._jobs[request.review_id] = job
        self._queue.put_nowait(job)

        return job

    async def run(self, request: ReviewRequest) -> ReviewJob:
        """Review one game here and now, without the queue.

        What a serverless job needs. RunPod hands a worker one job, waits for
        the call to return, and is free to shut the worker down the moment it
        does — so accepting a review and answering "queued" would kill the
        engine halfway through it. The internal queue exists for the long-lived
        box; here the platform's queue is the queue.
        """
        job = ReviewJob(request=request, total=len(request.moves))
        self._jobs[request.review_id] = job

        try:
            await self._run(job)
        except Exception as exc:  # noqa: BLE001 - reported, then re-raised
            log.exception("review %s failed", request.review_id)
            job.status = "failed"
            job.error = str(exc)
            job.finished_at = time.monotonic()
            # Laravel learns from the callback, not from the return value: the
            # player is watching a progress bar, and the platform's own record
            # of a failed job is not something the browser can see. Worth
            # insisting on for the same reason the final batch is — this one
            # is what gives the player their allowance back.
            await self._deliver_insistently(job, [], final=True)

            raise

        return job

    def job(self, review_id: str) -> ReviewJob | None:
        return self._jobs.get(review_id)

    def queue_depth(self) -> int:
        return self._queue.qsize()

    async def _run_forever(self) -> None:
        while True:
            job = await self._queue.get()

            try:
                await self._run(job)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad job must not stop the queue
                log.exception("review %s failed", job.request.review_id)
                job.status = "failed"
                job.error = str(exc)
                job.finished_at = time.monotonic()
                await self._deliver(job, [], final=True)
            finally:
                self._queue.task_done()

    async def _run(self, job: ReviewJob) -> None:
        request = job.request
        job.status = "running"
        job.started_at = time.monotonic()

        tracker = QueryTracker(
            query_id=request.review_id,
            requested_turns=list(range(len(request.moves))),
            max_turn=request.last_turn,
        )

        complete = asyncio.Event()
        pending: list[dict[str, Any]] = []

        def listener(response: Response) -> bool:
            if not tracker.accept(response):
                return False

            if tracker.is_complete:
                complete.set()

            return True

        # Before the engine is asked for anything.
        #
        # A review is delivered by calling Laravel back, so a review whose
        # callback cannot be made is minutes of a rented card spent producing an
        # answer with nowhere to go — and the old order of things only found
        # that out at the end, after paying for all of it. One empty batch up
        # front costs a round trip and turns that into an immediate failure.
        if not await self._deliver(job, []):
            raise RuntimeError(
                "could not reach Laravel to deliver this review, so it was not started"
            )

        self._engine.add_listener(listener)

        try:
            await self._engine.send(build_query(request))

            last_flush = time.monotonic()

            while not complete.is_set():
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(complete.wait(), timeout=POLL_INTERVAL_SECONDS)

                new = self._drain(tracker, job, pending)
                now = time.monotonic()

                should_flush = len(pending) >= BATCH_SIZE or (
                    pending and now - last_flush >= BATCH_INTERVAL_SECONDS
                )

                if should_flush:
                    # Only on success. Clearing regardless is what the comment
                    # below `_deliver` claimed was safe and was not: a batch
                    # that failed to send was dropped here and never resent, so
                    # a review could finish with a fifth of its moves missing,
                    # no error anywhere, and the search paid for in full.
                    if await self._deliver(job, pending):
                        pending.clear()

                    last_flush = now

                if new == 0 and not self._engine.is_running:
                    raise RuntimeError("the analysis engine stopped while reviewing")

            self._drain(tracker, job, pending)

            if tracker.error is not None:
                raise RuntimeError(tracker.error)

            job.status = "finished"
            job.finished_at = time.monotonic()

            # The one delivery worth retrying. It carries every report that has
            # not landed yet plus the flag that marks the review finished, and
            # by now the search has already been paid for — so a few seconds
            # spent on a second attempt is the cheapest thing in the whole job,
            # and losing it means the reconciler refunds a review that was in
            # fact completed.
            await self._deliver_insistently(job, pending, final=True)
            pending.clear()
        finally:
            self._engine.remove_listener(listener)

    @staticmethod
    def _drain(
        tracker: QueryTracker,
        job: ReviewJob,
        pending: list[dict[str, Any]],
    ) -> int:
        """Move newly arrived reports out of the tracker and into the outbox."""
        added = 0

        for turn, report in tracker.ordered_reports():
            if turn in job.reports:
                continue

            job.reports[turn] = report
            pending.append({"move_number": turn, "report": report})
            added += 1

        job.done = len(job.reports)

        return added

    async def _deliver_insistently(
        self,
        job: ReviewJob,
        reports: list[dict[str, Any]],
        *,
        final: bool = False,
        attempts: int = FINAL_DELIVERY_ATTEMPTS,
    ) -> bool:
        for attempt in range(attempts):
            if await self._deliver(job, reports, final=final):
                return True

            if attempt + 1 < attempts:
                # A deploy restarting, or a moment of packet loss. Long enough
                # to outlast one and short enough that a worker is not held open
                # waiting on a server that is simply gone.
                await asyncio.sleep(RETRY_BACKOFF_SECONDS * 2 ** attempt)

        return False

    async def _deliver(
        self,
        job: ReviewJob,
        reports: list[dict[str, Any]],
        *,
        final: bool = False,
    ) -> bool:
        """Send one batch. Returns whether Laravel took it.

        The caller keeps anything that did not land and sends it again with the
        next batch, so a callback that fails costs a retry rather than the moves
        it was carrying. Laravel upserts on `(review_id, move_number)`, which is
        what makes resending safe.
        """
        progress = job.snapshot()
        progress["final"] = final

        try:
            await self._sink(job.request.review_id, list(reports), progress)
        except Exception:  # noqa: BLE001 - a failed callback must not lose the review
            log.exception("could not deliver a batch for review %s", job.request.review_id)

            return False

        return True
