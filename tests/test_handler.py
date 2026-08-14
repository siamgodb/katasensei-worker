from __future__ import annotations

import pytest

from app import handler as rp
from app.config import Settings


class FakeEngine:
    """Stands in for the KataGo child process."""

    def __init__(self) -> None:
        self.is_running = True
        self.starts = 0
        self.stops = 0

    async def start(self) -> None:
        self.starts += 1

    async def stop(self) -> None:
        self.stops += 1
        self.is_running = False


class FakeReviewer:
    def __init__(self) -> None:
        self.ran: list = []

    async def run(self, request):
        self.ran.append(request)

        return _Job(request.review_id)


class FakeAnalyst:
    def __init__(self) -> None:
        self.asked: list = []

    async def analyse(self, request):
        self.asked.append(request)

        return {"winrate": 0.61, "to_play": "B"}


class FakeCallback:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _Job:
    def __init__(self, review_id: str) -> None:
        self.review_id = review_id

    def snapshot(self) -> dict:
        return {"review_id": self.review_id, "status": "finished"}


def settings(*, human_model: str | None = "/models/human.bin.gz") -> Settings:
    return Settings(
        katago_binary="katago",
        katago_model="/models/network.bin.gz",
        katago_config="/worker/configs/analysis-gpu.cfg",
        katago_human_model=human_model,
        laravel_url="https://play.siamgodb.com",
        callback_secret="secret",
        api_token="",
        default_visits=100,
        analysis_timeout_seconds=60.0,
    )


@pytest.fixture
def runtime(monkeypatch):
    """A started runtime, with the pieces below the engine faked out."""
    current = rp.Runtime(
        settings=settings(),
        engine=FakeEngine(),
        reviewer=FakeReviewer(),
        analyst=FakeAnalyst(),
        callback=FakeCallback(),
    )

    monkeypatch.setattr(rp, "_runtime", current)

    yield current

    monkeypatch.setattr(rp, "_runtime", None)


async def test_the_engine_is_started_once_and_kept(monkeypatch):
    """A cold start pays for the neural net. Every job after it must not."""
    engine = FakeEngine()

    monkeypatch.setattr(rp, "_runtime", None)
    monkeypatch.setattr(rp.Settings, "from_env", staticmethod(settings))
    monkeypatch.setattr(rp, "KataGoEngine", lambda **_: engine)

    first = await rp.runtime()
    second = await rp.runtime()

    assert first is second
    assert engine.starts == 1


async def test_a_dead_engine_is_replaced_rather_than_reused(monkeypatch):
    """Otherwise one crash poisons every job this worker is handed afterwards."""
    engines = [FakeEngine(), FakeEngine()]

    monkeypatch.setattr(rp, "_runtime", None)
    monkeypatch.setattr(rp.Settings, "from_env", staticmethod(settings))
    monkeypatch.setattr(rp, "KataGoEngine", lambda **_: engines.pop(0))

    first = await rp.runtime()
    first.engine.is_running = False

    second = await rp.runtime()

    assert second is not first
    assert first.engine.stops == 1
    assert second.engine.starts == 1


async def test_analyze_reaches_the_analyst(runtime):
    result = await rp.handler({
        "id": "rp-1",
        "input": {
            "kind": "analyze",
            "query_id": "req-1",
            "moves": [{"color": "B", "loc": "Q16"}],
            "board_x_size": 9,
            "board_y_size": 9,
            "max_visits": 400,
        },
    })

    assert result == {"winrate": 0.61, "to_play": "B"}

    asked = runtime.analyst.asked[0]
    assert asked.query_id == "req-1"
    assert asked.moves == [("B", "Q16")]
    assert asked.max_visits == 400


async def test_review_runs_to_completion_inside_the_job(runtime):
    """The return value is the proof it finished — RunPod may stop the worker
    the instant the handler returns, so answering 'queued' would kill it."""
    result = await rp.handler({
        "id": "rp-2",
        "input": {
            "kind": "review",
            "review_id": "01J",
            "moves": [{"color": "B", "loc": "Q16"}, {"color": "W", "loc": "D4"}],
            "max_visits": 600,
            "student_rank": "5k",
        },
    })

    assert result == {"review_id": "01J", "status": "finished"}

    ran = runtime.reviewer.ran[0]
    assert ran.review_id == "01J"
    assert ran.max_visits == 600
    assert ran.use_human_model is True


async def test_the_human_model_is_dropped_when_this_worker_has_none(monkeypatch, runtime):
    """A configuration question, not a request failure: the review still runs
    and comes back marked as degraded."""
    runtime.settings = settings(human_model=None)

    await rp.handler({
        "id": "rp-3",
        "input": {
            "kind": "review",
            "review_id": "01K",
            "moves": [{"color": "B", "loc": "Q16"}],
            "use_human_model": True,
        },
    })

    assert runtime.reviewer.ran[0].use_human_model is False


async def test_visits_fall_back_to_the_worker_default(runtime):
    await rp.handler({
        "id": "rp-4",
        "input": {
            "kind": "review",
            "review_id": "01L",
            "moves": [{"color": "B", "loc": "Q16"}],
            "max_visits": 0,
        },
    })

    assert runtime.reviewer.ran[0].max_visits == 100


async def test_an_unknown_kind_raises(runtime):
    """Raising is what puts the RunPod job into FAILED, which is the only thing
    the reconciler on the Laravel side can see."""
    with pytest.raises(ValueError, match="unknown kind"):
        await rp.handler({"id": "rp-5", "input": {"kind": "sharpen"}})


async def test_a_missing_input_is_not_a_crash(runtime):
    with pytest.raises(ValueError, match="unknown kind"):
        await rp.handler({"id": "rp-6"})


async def test_a_malformed_position_is_rejected_before_the_engine(runtime):
    """Validation lives in the schema both entry points share, so the thing on
    RunPod cannot accept what the development server refuses."""
    with pytest.raises(Exception):
        await rp.handler({
            "id": "rp-7",
            "input": {"kind": "analyze", "query_id": "req-2", "max_visits": 99999},
        })

    assert runtime.analyst.asked == []
