"""Reading KataGo's analysis-engine output.

The engine answers on stdout, one JSON object per line, and the answers to a
query do not arrive in order — several analysis threads work in parallel, and
teaching reports come on their own unordered stream. Everything here is about
sorting that back out, and none of it needs a running engine to test.

Two facts from ``docs/TeachingReport.md`` shape the whole design:

* A report for turn T is built from the searches at T *and* T+1, so asking for
  turn T implicitly asks for T+1 as well, and T+1's ordinary analysis response
  is emitted too.
* Reports arrive as separate messages tagged ``isTeachingReport``. They must be
  joined to their query on ``(id, turnNumber)`` and cannot be attached to
  either turn's analysis response.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterator


class ProtocolError(RuntimeError):
    """The engine said something we cannot make sense of."""


@dataclass(frozen=True)
class AnalysisResponse:
    """An ordinary position evaluation."""

    query_id: str
    turn_number: int
    payload: dict[str, Any]

    @property
    def is_final(self) -> bool:
        """False while the search is still reporting interim results."""
        return self.payload.get("isDuringSearch") is False


@dataclass(frozen=True)
class TeachingResponse:
    query_id: str
    turn_number: int
    report: dict[str, Any]


@dataclass(frozen=True)
class EngineError:
    query_id: str | None
    message: str


Response = AnalysisResponse | TeachingResponse | EngineError


def parse_line(line: str) -> Response | None:
    """Turn one line of engine output into a response.

    Returns ``None`` for blank lines and for warnings, which the engine emits
    alongside real answers and which are not addressed to any query.
    """
    line = line.strip()

    if not line:
        return None

    try:
        message = json.loads(line)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ProtocolError(f"engine emitted a non-JSON line: {line[:200]}") from exc

    if not isinstance(message, dict):
        raise ProtocolError(f"engine emitted a non-object line: {line[:200]}")

    query_id = message.get("id")

    if "error" in message:
        return EngineError(query_id=query_id, message=str(message["error"]))

    # Warnings accompany a successful answer; the answer itself is in the same
    # object, so this must not return early.
    if "warning" in message and "turnNumber" not in message:
        return None

    if message.get("isTeachingReport"):
        report = message.get("teachingReport")

        if not isinstance(report, dict):
            raise ProtocolError("teaching report message carried no report")

        return TeachingResponse(
            query_id=str(query_id),
            turn_number=int(message.get("turnNumber", report["position"]["turnNumber"])),
            report=report,
        )

    if "turnNumber" not in message:
        return None

    return AnalysisResponse(
        query_id=str(query_id),
        turn_number=int(message["turnNumber"]),
        payload=message,
    )


@dataclass
class QueryTracker:
    """Follows one whole-game query to completion.

    Completion is decided from the analysis responses, not the reports. The set
    of turns the engine will search is known in advance — the requested turns
    plus one past each of them — and a report is emitted the moment the second
    of its pair lands. So once every expected search has reported, every report
    that will ever arrive has arrived, including for positions that produce
    none.

    Counting reports instead would hang forever on a game whose last position is
    terminal, because that one gets no report.
    """

    query_id: str
    requested_turns: list[int]
    max_turn: int
    reports: dict[int, dict[str, Any]] = field(default_factory=dict)
    snapshots: dict[int, dict[str, Any]] = field(default_factory=dict)
    error: str | None = None

    @property
    def expected_turns(self) -> set[int]:
        turns: set[int] = set()

        for turn in self.requested_turns:
            turns.add(turn)

            # Asking for turn T implicitly asks for T+1, unless T is the last
            # position there is.
            if turn + 1 <= self.max_turn:
                turns.add(turn + 1)

        return turns

    @property
    def is_complete(self) -> bool:
        return self.error is not None or self.expected_turns <= set(self.snapshots)

    @property
    def progress(self) -> tuple[int, int]:
        return len(self.snapshots), len(self.expected_turns)

    def accept(self, response: Response) -> bool:
        """Take a response if it belongs to this query.

        Returns whether it was ours, so a dispatcher can route without knowing
        the shape of what it is routing.
        """
        if isinstance(response, EngineError):
            if response.query_id not in (None, self.query_id):
                return False

            self.error = response.message

            return True

        if response.query_id != self.query_id:
            return False

        if isinstance(response, TeachingResponse):
            self.reports[response.turn_number] = response.report
        elif response.is_final:
            self.snapshots[response.turn_number] = response.payload

        return True

    def ordered_reports(self) -> Iterator[tuple[int, dict[str, Any]]]:
        for turn in sorted(self.reports):
            yield turn, self.reports[turn]
