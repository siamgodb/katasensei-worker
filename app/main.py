"""The worker's HTTP face.

Deliberately small. Laravel asks for a review and gets an acknowledgement; the
results are pushed back as they are produced rather than polled for, because a
review takes minutes and holding a connection open for it helps nobody.
"""

from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager
from typing import Annotated, Any, AsyncIterator

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from .callbacks import LaravelCallback
from .config import Settings
from .engine import KataGoEngine
from .query import ReviewRequest
from .reviewer import Reviewer

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

    await engine.start()
    await reviewer.start()

    state.update(settings=settings, engine=engine, reviewer=reviewer, callback=callback)

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

    The worker is meant to sit on a private network with nothing else able to
    reach it. This is the second lock, not the first.
    """
    settings: Settings = state["settings"]
    expected = f"Bearer {settings.api_token}"

    if authorization is None or not secrets.compare_digest(authorization, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "bad token")


class MoveIn(BaseModel):
    color: str = Field(pattern="^[BW]$")
    loc: str


class ReviewIn(BaseModel):
    review_id: str
    moves: list[MoveIn]
    board_x_size: int = 19
    board_y_size: int = 19
    komi: float = 6.5
    rules: str = "japanese"
    initial_stones: list[MoveIn] = Field(default_factory=list)
    max_visits: int = 100
    student_rank: str | None = None
    student_color: str = "both"
    rank_gap: int = 3
    use_human_model: bool = True


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
            moves=[(m.color, m.loc) for m in body.moves],
            board_x_size=body.board_x_size,
            board_y_size=body.board_y_size,
            komi=body.komi,
            rules=body.rules,
            initial_stones=[(s.color, s.loc) for s in body.initial_stones] or None,
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
