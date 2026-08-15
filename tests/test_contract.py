"""The half of the boundary that lives on this side.

Reviews were rejected by this worker for as long as the review path existed.
Laravel sent `["B", "E5"]` pairs, the schema accepted only `{"color", "loc"}`
objects, and both test suites were green the whole time — Laravel's faked the
HTTP call and asserted the payload it had just built, and the worker's built
`ReviewRequest` objects directly and never parsed a payload at all. Each side
tested itself against itself.

These fixtures are the shared statement. Laravel's tests assert that what they
send equals these files; the tests here parse them with the real schemas and
check what comes out. Changing the shape on either side now fails the other
side's suite rather than production.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.query import build_position_query, build_query
from app.schemas import AnalyzeIn, MoveIn, ReviewIn

FIXTURES = Path(__file__).parent / "fixtures"


def payload(name: str) -> dict:
    body = json.loads((FIXTURES / name).read_text())
    body.pop("kind")

    return body


class TestWhatLaravelSendsForAReview:
    def test_it_parses(self) -> None:
        body = ReviewIn.model_validate(payload("laravel-review-request.json"))

        assert body.review_id == "01JAKESREVIEWID000000000"
        assert [m.as_pair() for m in body.moves] == [("W", "E5"), ("B", "pass")]
        assert [s.as_pair() for s in body.initial_stones] == [("B", "G7"), ("B", "C3")]
        assert body.board_x_size == 9
        assert body.student_rank == "4k"
        assert body.student_color == "white"
        assert body.use_human_model is False

    def test_it_reaches_the_engine_as_katago_expects(self) -> None:
        """The point of the whole exercise: what actually goes down the pipe."""
        from app.query import ReviewRequest

        body = ReviewIn.model_validate(payload("laravel-review-request.json"))

        query = build_query(
            ReviewRequest(
                review_id=body.review_id,
                moves=[m.as_pair() for m in body.moves],
                board_x_size=body.board_x_size,
                board_y_size=body.board_y_size,
                komi=body.komi,
                rules=body.rules,
                initial_stones=[s.as_pair() for s in body.initial_stones],
                max_visits=body.max_visits,
            )
        )

        assert query["moves"] == [["W", "E5"], ["B", "pass"]]
        assert query["initialStones"] == [["B", "G7"], ["B", "C3"]]


class TestWhatLaravelSendsForAPosition:
    def test_it_parses(self) -> None:
        body = AnalyzeIn.model_validate(payload("laravel-analyze-request.json"))

        assert body.query_id.startswith("live-")
        assert [m.as_pair() for m in body.moves] == [("W", "G3"), ("B", "pass")]
        assert [s.as_pair() for s in body.initial_stones] == [("B", "G7"), ("B", "C3")]
        assert body.max_visits == 400

    def test_it_reaches_the_engine_as_katago_expects(self) -> None:
        body = AnalyzeIn.model_validate(payload("laravel-analyze-request.json"))

        query = build_position_query(
            body.query_id,
            [m.as_pair() for m in body.moves],
            board_x_size=body.board_x_size,
            board_y_size=body.board_y_size,
            komi=body.komi,
            rules=body.rules,
            max_visits=body.max_visits,
            initial_stones=[s.as_pair() for s in body.initial_stones],
        )

        assert query["moves"] == [["W", "G3"], ["B", "pass"]]


class TestBothSpellingsOfAMove:
    def test_they_are_the_same_move(self) -> None:
        assert MoveIn.model_validate(["B", "Q16"]).as_pair() == ("B", "Q16")
        assert MoveIn.model_validate({"color": "B", "loc": "Q16"}).as_pair() == ("B", "Q16")

    def test_a_pair_of_the_wrong_length_is_refused(self) -> None:
        """Tolerant about which spelling, not about what a move is."""
        with pytest.raises(ValidationError, match="colour and a point"):
            MoveIn.model_validate(["B"])

        with pytest.raises(ValidationError, match="colour and a point"):
            MoveIn.model_validate(["B", "Q16", "extra"])

    def test_a_colour_that_is_not_a_colour_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            MoveIn.model_validate(["G", "Q16"])

        with pytest.raises(ValidationError):
            MoveIn.model_validate({"color": "black", "loc": "Q16"})
