"""One position, answered while somebody waits.

Reviews are queued because a whole game is minutes of work and nobody watches
it happen. The analysis board is the opposite: a person has just put a stone
down and is looking at the screen, so this bypasses the review queue entirely
and hands the query straight to the engine.

The engine parallelises across its own analysis threads, which is what makes
that safe to do while a review is running — the interactive query is answered
by a thread the review is not using, rather than waiting for the review to
finish. It does mean both go slightly slower, which is the right trade: a
review that takes eight minutes instead of seven is not noticed, and an
analysis that takes thirty seconds instead of three is abandoned.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from typing import Any

from .engine import KataGoEngine
from .protocol import AnalysisResponse, EngineError, Response
from .query import build_position_query

log = logging.getLogger(__name__)


class AnalysisFailed(RuntimeError):
    pass


@dataclass(frozen=True)
class PositionRequest:
    query_id: str
    moves: list[tuple[str, str]]
    board_x_size: int = 19
    board_y_size: int = 19
    komi: float = 6.5
    rules: str = "japanese"
    max_visits: int = 200
    initial_stones: list[tuple[str, str]] | None = None
    include_ownership: bool = True


class Analyst:
    def __init__(self, engine: KataGoEngine, timeout: float = 60.0) -> None:
        self._engine = engine
        self._timeout = timeout

    async def analyse(self, request: PositionRequest) -> dict[str, Any]:
        latest: dict[str, Any] = {}
        finished = asyncio.Event()
        failure: list[str] = []

        def listener(response: Response) -> bool:
            if isinstance(response, EngineError) and response.query_id == request.query_id:
                failure.append(response.message)
                finished.set()

                return True

            if not isinstance(response, AnalysisResponse):
                return False

            if response.query_id != request.query_id:
                return False

            # `reportDuringSearchEvery` means several of these arrive as the
            # search deepens. Each is a complete answer at the visits reached so
            # far, so keeping the last one is keeping the best one.
            latest.clear()
            latest.update(response.payload)

            if response.is_final:
                finished.set()

            return True

        self._engine.add_listener(listener)
        started = time.monotonic()

        try:
            await self._engine.send(build_position_query(
                request.query_id,
                request.moves,
                board_x_size=request.board_x_size,
                board_y_size=request.board_y_size,
                komi=request.komi,
                rules=request.rules,
                max_visits=request.max_visits,
                initial_stones=request.initial_stones,
            ))

            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(finished.wait(), timeout=self._timeout)
        finally:
            self._engine.remove_listener(listener)

        if failure:
            raise AnalysisFailed(failure[0])

        if not latest:
            raise AnalysisFailed("the engine did not answer in time")

        return summarise(
            latest,
            elapsed=time.monotonic() - started,
            include_ownership=request.include_ownership,
        )


def summarise(
    payload: dict[str, Any],
    *,
    elapsed: float = 0.0,
    include_ownership: bool = True,
    top: int = 6,
) -> dict[str, Any]:
    """Reduce an engine answer to what a board can draw.

    Everything is converted to Black's point of view here and nowhere else.
    KataGo reports winrate and score from the perspective of whoever is to
    move, which is a fine convention for an engine and a terrible one for a
    graph a human is reading — the bar would jump to the other side of the
    screen on every move while nothing actually changed.
    """
    root = payload.get("rootInfo") or {}
    to_play = str(root.get("currentPlayer", "B")).upper()
    flip = to_play == "W"

    def as_black(winrate: float, score: float) -> tuple[float, float]:
        return (1.0 - winrate, -score) if flip else (winrate, score)

    winrate, score_lead = as_black(
        float(root.get("winrate", 0.5)),
        float(root.get("scoreLead", 0.0)),
    )

    moves = []

    for info in sorted(payload.get("moveInfos", []), key=lambda m: m.get("order", 0))[:top]:
        move_winrate, move_score = as_black(
            float(info.get("winrate", 0.5)),
            float(info.get("scoreLead", 0.0)),
        )

        moves.append({
            "move": info.get("move"),
            "winrate": round(move_winrate, 4),
            "score_lead": round(move_score, 2),
            "visits": int(info.get("visits", 0)),
            "order": int(info.get("order", 0)),
            "prior": round(float(info.get("prior", 0.0)), 4),
            # Enough of the continuation to show what the engine has in mind,
            # not so much that a wrong guess ten moves deep is presented as
            # knowledge.
            "pv": list(info.get("pv", []))[:8],
        })

    summary: dict[str, Any] = {
        "turn": int(payload.get("turnNumber", 0)),
        "to_play": to_play,
        "winrate": round(winrate, 4),
        "score_lead": round(score_lead, 2),
        "visits": int(root.get("visits", 0)),
        "moves": moves,
        "elapsed_seconds": round(elapsed, 2),
    }

    if include_ownership and isinstance(payload.get("ownership"), list):
        # Also Black-positive, and rounded: two decimals is finer than any
        # shading a person can distinguish, and the array is 361 long.
        summary["ownership"] = [
            round(-value if flip else value, 2) for value in payload["ownership"]
        ]

    return summary
