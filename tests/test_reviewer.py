"""Delivering a review, including when Laravel will not take it.

A review is the expensive thing this worker does — minutes of a card that is
billed by the second — and it is the only thing whose result leaves by a route
nobody watches. Every failure here is silent by construction: the search
succeeds, RunPod records a completed job, and the moves are simply not there.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from app.query import ReviewRequest
from app.reviewer import Reviewer


class ScriptedEngine:
    """Answers a whole-game query with the responses KataGo would send.

    Emitted from a background task rather than inline, so the reviewer's loop
    runs the way it does in production: draining, flushing and re-checking
    while answers are still arriving.
    """

    def __init__(self, turns: int) -> None:
        self._turns = turns
        self._listeners: list = []
        self.is_running = True
        self.sent: list[dict[str, Any]] = []

    def add_listener(self, listener) -> None:
        self._listeners.append(listener)

    def remove_listener(self, listener) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    async def send(self, query: dict[str, Any]) -> None:
        self.sent.append(query)
        asyncio.get_running_loop().call_soon(self._replay, query["id"])

    def _replay(self, query_id: str) -> None:
        from app.protocol import AnalysisResponse, TeachingResponse

        for turn in range(self._turns):
            self._dispatch(TeachingResponse(
                query_id=query_id,
                turn_number=turn,
                report={"position": {"turnNumber": turn, "mover": "B"}},
            ))

        for turn in range(self._turns):
            self._dispatch(AnalysisResponse(
                query_id=query_id,
                turn_number=turn,
                payload={"turnNumber": turn, "isDuringSearch": False},
            ))

    def _dispatch(self, response) -> None:
        for listener in list(self._listeners):
            if listener(response):
                return


class Sink:
    """A callback that can be told to fail, and remembers what it was given."""

    def __init__(self, *, fail_until: int = 0) -> None:
        self.calls: list[tuple[str, list[dict], dict]] = []
        self._fail_until = fail_until

    async def __call__(self, review_id: str, reports: list, progress: dict) -> None:
        self.calls.append((review_id, reports, progress))

        if len(self.calls) <= self._fail_until:
            raise RuntimeError("laravel said no")

    @property
    def delivered_turns(self) -> set[int]:
        """Every move number Laravel actually received, over every batch."""
        return {r["move_number"] for _, reports, _ in self.calls for r in reports}


def request(turns: int = 3) -> ReviewRequest:
    return ReviewRequest(
        review_id="01JREVIEW",
        moves=[("B", "Q16")] * turns,
        max_visits=100,
    )


async def test_a_review_that_cannot_be_delivered_is_not_started(monkeypatch):
    """The whole point: find out before the card is paid for, not after.

    A review is only worth running if its result can be sent somewhere. The
    handshake is one empty batch, and its failure has to stop the job before
    the engine is asked for anything.
    """
    monkeypatch.setattr("app.reviewer.RETRY_BACKOFF_SECONDS", 0.0)

    engine = ScriptedEngine(turns=3)
    reviewer = Reviewer(engine, Sink(fail_until=99))

    with pytest.raises(RuntimeError, match="could not reach Laravel"):
        await reviewer.run(request())

    assert engine.sent == [], "the engine was asked to search anyway"


async def test_a_failed_batch_is_kept_and_sent_again(monkeypatch):
    """The bug this file was written for.

    Batches used to be cleared whether or not they landed, so a single failed
    callback lost the moves it was carrying for good: a finished review, a
    completed RunPod job, a fifth of the game missing and no error anywhere.
    """
    monkeypatch.setattr("app.reviewer.BATCH_SIZE", 1)

    engine = ScriptedEngine(turns=3)
    # The handshake goes through; the first batch of reports does not.
    sink = Sink()
    reviewer = Reviewer(engine, sink)

    async def fail_the_second_call(review_id, reports, progress):
        sink.calls.append((review_id, reports, progress))

        if len(sink.calls) == 2:
            raise RuntimeError("laravel said no")

    reviewer._sink = fail_the_second_call

    job = await asyncio.wait_for(reviewer.run(request()), timeout=5)

    assert job.status == "finished"
    assert sink.delivered_turns == {0, 1, 2}, "a batch was dropped rather than resent"


async def test_every_report_reaches_laravel_exactly_once_when_nothing_fails():
    engine = ScriptedEngine(turns=3)
    sink = Sink()

    job = await asyncio.wait_for(Reviewer(engine, sink).run(request()), timeout=5)

    assert job.status == "finished"
    assert sink.delivered_turns == {0, 1, 2}

    delivered = [r["move_number"] for _, reports, _ in sink.calls for r in reports]
    assert sorted(delivered) == delivered
    assert len(delivered) == 3, "a report was sent twice on the happy path"


async def test_the_last_batch_is_retried_before_the_review_is_given_up_on(monkeypatch):
    """It carries the flag that marks the review finished.

    By the time it is sent the search has already been paid for, so a second
    attempt is the cheapest thing in the job — and without one the reconciler
    refunds a review that in fact completed.
    """
    monkeypatch.setattr("app.reviewer.RETRY_BACKOFF_SECONDS", 0.0)

    engine = ScriptedEngine(turns=2)
    sink = Sink()
    reviewer = Reviewer(engine, sink)

    calls = {"n": 0}

    async def fail_the_first_final_call(review_id, reports, progress):
        sink.calls.append((review_id, reports, progress))

        if progress.get("final"):
            calls["n"] += 1

            if calls["n"] == 1:
                raise RuntimeError("laravel restarted mid-deploy")

    reviewer._sink = fail_the_first_final_call

    job = await asyncio.wait_for(reviewer.run(request(turns=2)), timeout=5)

    assert job.status == "finished"
    assert calls["n"] == 2, "the final batch was sent once and abandoned"

    final = [progress for _, _, progress in sink.calls if progress.get("final")]
    assert final[-1]["status"] == "finished"


async def test_a_failure_is_reported_to_laravel_before_it_is_re_raised(monkeypatch):
    """Laravel learns from the callback; RunPod's record is not something the
    browser can see, and the player is watching a progress bar."""
    monkeypatch.setattr("app.reviewer.POLL_INTERVAL_SECONDS", 0.01)

    engine = ScriptedEngine(turns=1)
    engine.is_running = False
    sink = Sink()

    async def send_nothing(query):
        engine.sent.append(query)

    engine.send = send_nothing

    with pytest.raises(RuntimeError, match="stopped while reviewing"):
        await asyncio.wait_for(Reviewer(engine, sink).run(request(turns=1)), timeout=5)

    final = [progress for _, _, progress in sink.calls if progress.get("final")]
    assert final and final[-1]["status"] == "failed"


async def test_the_progress_it_reports_is_json():
    """It goes into an HTTP body. A float that is not finite is not JSON, and
    the failure would be at the callback rather than anywhere useful."""
    engine = ScriptedEngine(turns=2)
    sink = Sink()

    await asyncio.wait_for(Reviewer(engine, sink).run(request(turns=2)), timeout=5)

    for _, reports, progress in sink.calls:
        json.dumps({"reports": reports, "progress": progress}, allow_nan=False)
