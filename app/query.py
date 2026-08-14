"""Building the analysis query for a whole game.

One query per game, never one per move. A report for turn T is built from the
searches at T and T+1, and the position after T is the position before T+1, so
a sequential walk over N moves costs N+1 searches rather than 2N. Splitting the
game up doubles the work for nothing — which on a CPU box is the difference
between an eight-minute review and a sixteen-minute one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReviewRequest:
    """What Laravel asks for."""

    review_id: str
    moves: list[tuple[str, str]]
    board_x_size: int = 19
    board_y_size: int = 19
    komi: float = 6.5
    rules: str = "japanese"
    initial_stones: list[tuple[str, str]] | None = None
    max_visits: int = 100
    student_rank: str | None = None
    student_color: str = "both"
    rank_gap: int = 3
    # Off for the free tier. The human model is an 18-block net and is the
    # single most expensive thing in a CPU review; without it the reports come
    # back with humanComparison null and degraded ["NO_HUMAN_MODEL"], which the
    # fact sheet already handles.
    use_human_model: bool = True

    @property
    def last_turn(self) -> int:
        return max(len(self.moves) - 1, 0)


def build_query(request: ReviewRequest) -> dict[str, Any]:
    """The JSON the engine is given, as one line on stdin."""
    turns = list(range(len(request.moves)))

    query: dict[str, Any] = {
        "id": request.review_id,
        "moves": [list(move) for move in request.moves],
        "boardXSize": request.board_x_size,
        "boardYSize": request.board_y_size,
        "komi": request.komi,
        "rules": request.rules,
        "analyzeTurns": turns,
        "maxVisits": request.max_visits,
        # Forced on by the engine anyway when teaching reports are requested,
        # since the reports are built from them. Stated here so the query says
        # what it means.
        "includeOwnership": True,
        "includePolicy": True,
        "includeTeachingReport": True,
        "teachingStudentColor": request.student_color,
        "teachingRankGap": request.rank_gap,
    }

    if request.initial_stones:
        query["initialStones"] = [list(stone) for stone in request.initial_stones]

    if request.student_rank and request.use_human_model:
        query["teachingStudentRank"] = request.student_rank
        # The human policy is only evaluated at a profile the search knows
        # about, so the profile has to be set for the query as well as named
        # for the report.
        query["overrideSettings"] = {"humanSLProfile": f"rank_{request.student_rank}"}

    return query


def build_position_query(
    query_id: str,
    moves: list[tuple[str, str]],
    *,
    board_x_size: int = 19,
    board_y_size: int = 19,
    komi: float = 6.5,
    rules: str = "japanese",
    max_visits: int = 200,
    initial_stones: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """A single position, for the interactive analysis board.

    No teaching report: the analysis board wants candidate moves and ownership
    as fast as they can be produced, and a report needs a second search at the
    following position that nobody has played yet.
    """
    query: dict[str, Any] = {
        "id": query_id,
        "moves": [list(move) for move in moves],
        "boardXSize": board_x_size,
        "boardYSize": board_y_size,
        "komi": komi,
        "rules": rules,
        "maxVisits": max_visits,
        "includeOwnership": True,
        "includePolicy": True,
        # Interim results as the search deepens, so the board can show
        # something within a second instead of nothing for thirty.
        "reportDuringSearchEvery": 0.5,
    }

    if initial_stones:
        query["initialStones"] = [list(stone) for stone in initial_stones]

    return query
