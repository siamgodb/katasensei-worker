"""What a caller may ask for.

Shared by both entry points — the RunPod handler and the FastAPI app — so that
the thing running in production and the thing running on a laptop cannot drift
apart in what they accept.

That was never the drift that mattered. The two callers on the Laravel side had
each picked their own move shape — the analysis board sending `{color, loc}`,
reviews sending `["B", "E5"]` — and only one of them could be right. Reviews
had been sending a shape this file refuses since the day it was written, and
neither test suite noticed: Laravel's faked the HTTP call and the worker's built
its own moves, so both sides passed while agreeing on nothing.

`tests/test_contract.py` is what stops that happening again. It parses the exact
payloads Laravel produces, from fixtures Laravel's own tests assert against.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class MoveIn(BaseModel):
    """A colour and a point, written either way round.

    `["B", "Q16"]` is KataGo's own wire format and what the analysis board
    sends; `{"color": "B", "loc": "Q16"}` is what the review path sends. Both
    are accepted and normalise to the same thing.

    Taking one and refusing the other is what was here before, and it is how
    reviews spent months being rejected by a worker that had never been asked
    to parse a real Laravel payload. A boundary this narrow is not worth a
    production failure, and both shapes are unambiguous.
    """

    color: str = Field(pattern="^[BW]$")
    loc: str
    """A point in the letter-number form the engine speaks, or "pass" — which is
    a position it has an opinion about like any other."""

    @model_validator(mode="before")
    @classmethod
    def _also_accept_a_pair(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise ValueError(
                    f"a move written as a pair needs a colour and a point, got {value!r}"
                )

            return {"color": value[0], "loc": value[1]}

        return value

    def as_pair(self) -> tuple[str, str]:
        """What everything below this layer holds, and what KataGo is sent."""
        return (self.color, self.loc)


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
    """A whole game, to be walked move by move.

    The bounds are a spending limit, not input hygiene. This is the one request
    that can occupy a rented card for minutes at a time, and its cost is very
    nearly `len(moves) * max_visits` — so an unbounded pair of numbers here is
    an unbounded bill, reachable by one wrong migration on the Laravel side. The
    ceilings are set above what the most expensive plan asks for (600 visits, and
    a 19x19 game that runs long), so nothing legitimate meets them.
    """

    review_id: str
    moves: list[MoveIn] = Field(max_length=1000)
    board_x_size: int = 19
    board_y_size: int = 19
    komi: float = 6.5
    rules: str = "japanese"
    initial_stones: list[MoveIn] = Field(default_factory=list)
    # Zero means "whatever this worker's default is", which the handler applies.
    max_visits: int = Field(default=100, ge=0, le=4000)
    student_rank: str | None = None
    student_color: str = "both"
    rank_gap: int = 3
    use_human_model: bool = True
