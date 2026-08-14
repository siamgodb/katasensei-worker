"""Recorded engine output, replayed.

KataGo is not run here — building it takes a compiler toolchain and several
hundred megabytes of neural net. Everything below is the part that would still
be wrong if it were: which message belongs to which query, when a query is
finished, and what to do with the ones that arrive out of order.
"""

from __future__ import annotations

import json

import pytest

from app.protocol import (
    AnalysisResponse,
    EngineError,
    ProtocolError,
    QueryTracker,
    TeachingResponse,
    parse_line,
)


def analysis(query_id: str, turn: int, *, during_search: bool = False) -> str:
    return json.dumps(
        {
            "id": query_id,
            "turnNumber": turn,
            "isDuringSearch": during_search,
            "rootInfo": {"winrate": 0.51, "scoreLead": -1.2, "visits": 100},
            "moveInfos": [{"move": "Q16", "order": 0, "visits": 60, "scoreLead": -1.0}],
        }
    )


def teaching(query_id: str, turn: int) -> str:
    return json.dumps(
        {
            "id": query_id,
            "turnNumber": turn,
            "isTeachingReport": True,
            "teachingReport": {
                "position": {"turnNumber": turn, "mover": "B", "playedMove": "Q16"},
                "moveEval": {"severity": "GOOD", "pointsLost": 0.2},
            },
        }
    )


class TestParsing:
    def test_reads_an_analysis_response(self) -> None:
        response = parse_line(analysis("r1", 7))

        assert isinstance(response, AnalysisResponse)
        assert response.query_id == "r1"
        assert response.turn_number == 7
        assert response.is_final

    def test_marks_an_interim_result_as_not_final(self) -> None:
        # The engine reports as the search deepens. Treating an interim result
        # as the answer would finish a review on a shallow read of every
        # position.
        response = parse_line(analysis("r1", 7, during_search=True))

        assert isinstance(response, AnalysisResponse)
        assert not response.is_final

    def test_reads_a_teaching_report(self) -> None:
        response = parse_line(teaching("r1", 3))

        assert isinstance(response, TeachingResponse)
        assert response.turn_number == 3
        assert response.report["moveEval"]["severity"] == "GOOD"

    def test_reads_an_error(self) -> None:
        response = parse_line(json.dumps({"id": "r1", "error": "Could not parse rules"}))

        assert isinstance(response, EngineError)
        assert "rules" in response.message

    def test_ignores_blank_lines(self) -> None:
        assert parse_line("") is None
        assert parse_line("   \n") is None

    def test_ignores_a_bare_warning(self) -> None:
        # Warnings with no answer attached are not addressed to any query.
        assert parse_line(json.dumps({"warning": "Uncommon board size", "id": "r1"})) is None

    def test_keeps_an_answer_that_came_with_a_warning(self) -> None:
        line = json.dumps(
            {
                "id": "r1",
                "turnNumber": 2,
                "isDuringSearch": False,
                "warning": "maxVisits is very low",
                "rootInfo": {},
            }
        )

        response = parse_line(line)

        assert isinstance(response, AnalysisResponse)
        assert response.turn_number == 2

    def test_rejects_output_that_is_not_json(self) -> None:
        with pytest.raises(ProtocolError):
            parse_line("Segmentation fault")

    def test_rejects_a_report_message_with_no_report(self) -> None:
        with pytest.raises(ProtocolError):
            parse_line(json.dumps({"id": "r1", "turnNumber": 1, "isTeachingReport": True}))


class TestQueryTracker:
    def tracker(self, moves: int = 3) -> QueryTracker:
        return QueryTracker(
            query_id="r1",
            requested_turns=list(range(moves)),
            max_turn=moves - 1,
        )

    def test_expects_one_search_past_the_last_requested_turn(self) -> None:
        # A report for turn T needs the searches at T and T+1, so asking for
        # turns 0-2 of a three-move game means searching 0, 1 and 2 — the
        # extension is clamped at the last position that exists.
        assert self.tracker(3).expected_turns == {0, 1, 2}

    def test_extends_the_turn_set_for_a_partial_range(self) -> None:
        tracker = QueryTracker(query_id="r1", requested_turns=[5, 6], max_turn=200)

        assert tracker.expected_turns == {5, 6, 7}

    def test_completes_on_the_searches_not_the_reports(self) -> None:
        # Counting reports would hang forever on a game whose last position is
        # terminal: that one produces no report, and the query would never be
        # considered finished.
        tracker = self.tracker(3)

        for turn in (0, 1, 2):
            tracker.accept(parse_line(analysis("r1", turn)))

        assert tracker.is_complete
        assert tracker.reports == {}

    def test_stays_incomplete_while_a_search_is_missing(self) -> None:
        tracker = self.tracker(3)

        tracker.accept(parse_line(analysis("r1", 0)))
        tracker.accept(parse_line(analysis("r1", 2)))

        assert not tracker.is_complete

    def test_ignores_interim_results(self) -> None:
        tracker = self.tracker(1)

        tracker.accept(parse_line(analysis("r1", 0, during_search=True)))
        assert not tracker.is_complete

        tracker.accept(parse_line(analysis("r1", 0)))
        assert tracker.is_complete

    def test_collects_reports_arriving_out_of_order(self) -> None:
        # Several analysis threads finish in whatever order they finish, and
        # reports come on a stream of their own.
        tracker = self.tracker(3)

        for line in (
            teaching("r1", 2),
            analysis("r1", 1),
            teaching("r1", 0),
            analysis("r1", 2),
            teaching("r1", 1),
            analysis("r1", 0),
        ):
            tracker.accept(parse_line(line))

        assert tracker.is_complete
        assert [turn for turn, _ in tracker.ordered_reports()] == [0, 1, 2]

    def test_refuses_another_query_s_messages(self) -> None:
        tracker = self.tracker(3)

        assert tracker.accept(parse_line(analysis("somebody-else", 0))) is False
        assert tracker.snapshots == {}

    def test_an_error_finishes_the_query(self) -> None:
        tracker = self.tracker(3)

        tracker.accept(parse_line(json.dumps({"id": "r1", "error": "out of memory"})))

        assert tracker.is_complete
        assert tracker.error == "out of memory"

    def test_takes_an_error_that_names_no_query(self) -> None:
        # A fatal engine error has no id. Ignoring it would leave the review
        # waiting for searches that will never come.
        tracker = self.tracker(3)

        tracker.accept(parse_line(json.dumps({"error": "model file is corrupt"})))

        assert tracker.error == "model file is corrupt"

    def test_reports_progress(self) -> None:
        tracker = self.tracker(4)

        tracker.accept(parse_line(analysis("r1", 0)))
        tracker.accept(parse_line(analysis("r1", 1)))

        assert tracker.progress == (2, 4)
