"""The worker's HTTP face, for running it without RunPod.

Production is `app.handler` on a RunPod serverless endpoint. This is the same
engine, reviewer and analyst behind an ordinary HTTP server: what you run on a
laptop or on a box you own, and what the tests exercise. Both entry points take
the same request shapes, from `app.schemas`, so the two cannot drift.

Reviews here go through the worker's own queue and answer immediately, because
this process is expected to outlive the request. On RunPod it cannot, which is
the one real difference between the two.
"""

from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager
from typing import Annotated, Any, AsyncIterator

from fastapi import Depends, FastAPI, Header, HTTPException, status

from .analysis import AnalysisFailed, Analyst, PositionRequest
from .callbacks import LaravelCallback
from .config import Settings
from .engine import KataGoEngine
from .query import ReviewRequest
from .reviewer import Reviewer
from .schemas import AnalyzeIn, ReviewIn

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings.from_env()

    engine = KataGoEngine(
        binary=settings.katago_binary,
        model=settings.katago_model,
        config=settings.katago_config,
        human_model=settings.katago_human_model,
    )

    callback = LaravelCallback(settings.laravel_url, settings.callback_secret)
    reviewer = Reviewer(engine, callback.send_batch)
    analyst = Analyst(engine, timeout=settings.analysis_timeout_seconds)

    await engine.start()
    await reviewer.start()

    state.update(
        settings=settings,
        engine=engine,
        reviewer=reviewer,
        callback=callback,
        analyst=analyst,
    )

    try:
        yield
    finally:
        await reviewer.stop()
        await engine.stop()
        await callback.close()
        state.clear()


app = FastAPI(title="katasensei-worker", lifespan=lifespan)


def authorise(authorization: Annotated[str | None, Header()] = None) -> None:
    """A shared bearer token, compared in constant time.

    Unset means this face is closed. It is the development entry point and has
    no equivalent of RunPod's account-level key in front of it, so a missing
    token has to mean "nobody" rather than "anybody".
    """
    settings: Settings = state["settings"]

    if settings.api_token == "":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "no API_TOKEN is set")

    expected = f"Bearer {settings.api_token}"

    if authorization is None or not secrets.compare_digest(authorization, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "bad token")


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    engine: KataGoEngine = state["engine"]
    reviewer: Reviewer = state["reviewer"]

    return {
        "engine_running": engine.is_running,
        "queue_depth": reviewer.queue_depth(),
    }


@app.post("/v1/review", status_code=status.HTTP_202_ACCEPTED)
async def start_review(
    body: ReviewIn,
    _: Annotated[None, Depends(authorise)],
) -> dict[str, Any]:
    reviewer: Reviewer = state["reviewer"]
    settings: Settings = state["settings"]

    if reviewer.job(body.review_id) is not None:
        # Laravel retried, or two workers picked up the same job. Answering with
        # the existing job is idempotent; starting a second one would review the
        # game twice and bill for it once.
        return reviewer.job(body.review_id).snapshot()  # type: ignore[union-attr]

    job = reviewer.submit(
        ReviewRequest(
            review_id=body.review_id,
            moves=[m.as_pair() for m in body.moves],
            board_x_size=body.board_x_size,
            board_y_size=body.board_y_size,
            komi=body.komi,
            rules=body.rules,
            initial_stones=[s.as_pair() for s in body.initial_stones] or None,
            max_visits=body.max_visits or settings.default_visits,
            student_rank=body.student_rank,
            student_color=body.student_color,
            rank_gap=body.rank_gap,
            # Asking for the human model when the box has not been given one is
            # a configuration question, not a request failure: the review runs
            # and comes back marked as degraded.
            use_human_model=body.use_human_model and settings.katago_human_model is not None,
        )
    )

    return job.snapshot()


@app.get("/v1/review/{review_id}")
async def review_status(
    review_id: str,
    _: Annotated[None, Depends(authorise)],
) -> dict[str, Any]:
    reviewer: Reviewer = state["reviewer"]
    job = reviewer.job(review_id)

    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such review")

    return job.snapshot()


@app.post("/v1/analyze")
async def analyze(
    body: AnalyzeIn,
    _: Annotated[None, Depends(authorise)],
) -> dict[str, Any]:
    """Answer a single position, synchronously.

    The caller is a queued Laravel job with a person waiting behind it, so the
    connection is held open rather than answered with a callback: the whole
    thing is seconds, and a callback for something this short is two more
    moving parts to get wrong.
    """
    analyst: Analyst = state["analyst"]

    try:
        return await analyst.analyse(
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
    except AnalysisFailed as exc:
        # 503 rather than 500: the engine is busy or wedged, and the caller
        # should try again rather than treat the position as unanalysable.
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
