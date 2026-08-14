"""What a caller may ask for.

Shared by both entry points — the RunPod handler and the FastAPI app — so that
the thing running in production and the thing running on a laptop cannot drift
apart in what they accept.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class MoveIn(BaseModel):
    color: str = Field(pattern="^[BW]$")
    loc: str


class AnalyzeIn(BaseModel):
    """One position from the analysis board."""

    query_id: str
    moves: list[MoveIn] = Field(default_factory=list)
    initial_stones: list[MoveIn] = Field(default_factory=list)
    board_x_size: int = 19
    board_y_size: int = 19
    komi: float = 6.5
    rules: str = "japanese"
    # Bounded here as well as in Laravel. This endpoint spends GPU seconds that
    # are billed by the second, and the bound is the only thing between a typo
    # and a request that occupies a worker for an hour.
    max_visits: int = Field(default=200, ge=10, le=4000)
    include_ownership: bool = True


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
