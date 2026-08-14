from __future__ import annotations

import asyncio

import pytest

from app.analysis import AnalysisFailed, Analyst, PositionRequest, summarise
from app.protocol import AnalysisResponse, EngineError


class FakeEngine:
    """Accepts a query and replays canned responses to whoever is listening."""

    def __init__(self, responses: list[object]) -> None:
        self._responses = responses
        self._listeners: list = []
        self.sent: list[dict] = []
        self.is_running = True

    def add_listener(self, listener) -> None:
        self._listeners.append(listener)

    def remove_listener(self, listener) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    async def send(self, query: dict) -> None:
        self.sent.append(query)

        for response in self._responses:
            for listener in list(self._listeners):
                if listener(response):
                    break


def payload(*, turn=10, player="B", winrate=0.62, score=3.4, during=False, ownership=None):
    body = {
        "id": "live-1",
        "turnNumber": turn,
        "isDuringSearch": during,
        "rootInfo": {
            "currentPlayer": player,
            "winrate": winrate,
            "scoreLead": score,
            "visits": 200,
        },
        "moveInfos": [
            {"move": "Q16", "order": 0, "winrate": winrate, "scoreLead": score,
             "visits": 120, "prior": 0.31, "pv": ["Q16", "D4", "Q4"]},
            {"move": "D16", "order": 1, "winrate": winrate - 0.05, "scoreLead": score - 1.5,
             "visits": 40, "prior": 0.12, "pv": ["D16"]},
        ],
    }

    if ownership is not None:
        body["ownership"] = ownership

    return body


class TestSummarise:
    def test_keeps_black_perspective_when_black_is_to_play(self):
        summary = summarise(payload(player="B", winrate=0.62, score=3.4))

        assert summary["winrate"] == 0.62
        assert summary["score_lead"] == 3.4
        assert summary["to_play"] == "B"

    def test_flips_the_engines_perspective_when_white_is_to_play(self):
        # KataGo answers from the point of view of whoever is to move. Left
        # alone, the bar would jump to the other side of the screen every move
        # while nothing actually changed.
        summary = summarise(payload(player="W", winrate=0.62, score=3.4))

        assert summary["winrate"] == pytest.approx(0.38)
        assert summary["score_lead"] == pytest.approx(-3.4)

    def test_flips_candidate_moves_and_ownership_too(self):
        summary = summarise(payload(player="W", ownership=[0.5, -0.25]))

        assert summary["moves"][0]["winrate"] == pytest.approx(0.38)
        assert summary["moves"][0]["score_lead"] == pytest.approx(-3.4)
        assert summary["ownership"] == [-0.5, 0.25]

    def test_orders_candidates_and_trims_the_principal_variation(self):
        body = payload()
        body["moveInfos"][0]["pv"] = [f"A{i}" for i in range(20)]
        body["moveInfos"].reverse()

        summary = summarise(body)

        assert [move["move"] for move in summary["moves"]] == ["Q16", "D16"]
        assert len(summary["moves"][0]["pv"]) == 8

    def test_leaves_ownership_out_when_not_wanted(self):
        summary = summarise(payload(ownership=[0.5]), include_ownership=False)

        assert "ownership" not in summary


class TestAnalyst:
    def test_returns_the_deepest_answer(self):
        engine = FakeEngine([
            AnalysisResponse("live-1", 10, payload(winrate=0.5, during=True)),
            AnalysisResponse("live-1", 10, payload(winrate=0.62, during=False)),
        ])

        summary = asyncio.run(
            Analyst(engine).analyse(PositionRequest(query_id="live-1", moves=[("B", "Q16")]))
        )

        # Interim answers arrive as the search deepens; the last one is the one
        # with the most visits behind it.
        assert summary["winrate"] == 0.62

    def test_ignores_answers_to_somebody_elses_query(self):
        engine = FakeEngine([
            AnalysisResponse("a-review", 3, payload(winrate=0.1)),
            AnalysisResponse("live-1", 10, payload(winrate=0.62)),
        ])

        summary = asyncio.run(
            Analyst(engine).analyse(PositionRequest(query_id="live-1", moves=[]))
        )

        assert summary["winrate"] == 0.62

    def test_raises_when_the_engine_refuses_the_position(self):
        engine = FakeEngine([EngineError("live-1", "illegal move")])

        with pytest.raises(AnalysisFailed, match="illegal move"):
            asyncio.run(Analyst(engine).analyse(PositionRequest(query_id="live-1", moves=[])))

    def test_raises_when_nothing_comes_back(self):
        engine = FakeEngine([])

        with pytest.raises(AnalysisFailed):
            asyncio.run(
                Analyst(engine, timeout=0.05).analyse(
                    PositionRequest(query_id="live-1", moves=[])
                )
            )

    def test_passes_the_position_to_the_engine(self):
        engine = FakeEngine([AnalysisResponse("live-1", 1, payload())])

        asyncio.run(
            Analyst(engine).analyse(
                PositionRequest(
                    query_id="live-1",
                    moves=[("B", "Q16")],
                    board_x_size=13,
                    board_y_size=13,
                    komi=7.5,
                    max_visits=64,
                )
            )
        )

        query = engine.sent[0]

        assert query["moves"] == [["B", "Q16"]]
        assert query["boardXSize"] == 13
        assert query["komi"] == 7.5
        assert query["maxVisits"] == 64
