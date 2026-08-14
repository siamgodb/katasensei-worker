from __future__ import annotations

from app.query import ReviewRequest, build_position_query, build_query


def request(**overrides) -> ReviewRequest:
    defaults = dict(
        review_id="rev-1",
        moves=[("B", "Q16"), ("W", "D4"), ("B", "Q4")],
        student_rank="5k",
    )

    return ReviewRequest(**{**defaults, **overrides})


class TestReviewQuery:
    def test_asks_for_the_whole_game_in_one_query(self) -> None:
        # One query per game, never one per move: the position after turn T is
        # the position before T+1, so a sequential walk costs N+1 searches
        # rather than 2N. On a CPU box that is the difference between an
        # eight-minute review and a sixteen-minute one.
        query = build_query(request())

        assert query["analyzeTurns"] == [0, 1, 2]
        assert query["id"] == "rev-1"
        assert query["moves"] == [["B", "Q16"], ["W", "D4"], ["B", "Q4"]]

    def test_asks_for_teaching_reports(self) -> None:
        query = build_query(request())

        assert query["includeTeachingReport"] is True
        assert query["includeOwnership"] is True
        assert query["includePolicy"] is True

    def test_names_the_student_and_sets_the_matching_profile(self) -> None:
        # The human policy is only evaluated at a profile the search knows
        # about, so naming the rank in the report request is not enough on its
        # own.
        query = build_query(request(student_rank="2d"))

        assert query["teachingStudentRank"] == "2d"
        assert query["overrideSettings"]["humanSLProfile"] == "rank_2d"

    def test_leaves_the_rank_out_when_the_human_model_is_off(self) -> None:
        # The free tier. The reports come back with humanComparison null, which
        # the fact sheet handles by dropping the rank comparison and keeping
        # everything else.
        query = build_query(request(use_human_model=False))

        assert "teachingStudentRank" not in query
        assert "overrideSettings" not in query

    def test_leaves_the_rank_out_when_it_is_unknown(self) -> None:
        query = build_query(request(student_rank=None))

        assert "teachingStudentRank" not in query

    def test_carries_handicap_stones(self) -> None:
        query = build_query(request(initial_stones=[("B", "Q16"), ("B", "D4")]))

        assert query["initialStones"] == [["B", "Q16"], ["B", "D4"]]

    def test_omits_initial_stones_for_an_even_game(self) -> None:
        assert "initialStones" not in build_query(request())

    def test_passes_the_tier_s_visit_count_through(self) -> None:
        assert build_query(request(max_visits=600))["maxVisits"] == 600

    def test_handles_a_game_with_no_moves(self) -> None:
        query = build_query(request(moves=[]))

        assert query["analyzeTurns"] == []


class TestPositionQuery:
    def test_asks_for_interim_results(self) -> None:
        # The analysis board should show something within a second rather than
        # nothing for thirty, which on a CPU box is the difference between
        # usable and not.
        query = build_position_query("live-1", [("B", "Q16")])

        assert query["reportDuringSearchEvery"] == 0.5

    def test_asks_for_no_teaching_report(self) -> None:
        # A report needs a second search at the following position, which on an
        # analysis board nobody has played yet.
        query = build_position_query("live-1", [("B", "Q16")])

        assert "includeTeachingReport" not in query
        assert "analyzeTurns" not in query
