from __future__ import annotations

import time

import pytest

from app.engine import EngineNotRunning, KataGoEngine


class ScriptedEngine(KataGoEngine):
    """A KataGo that behaves however the script says, then exits.

    Everything above `command()` is the real thing — the same start, the same
    readiness wait, the same stderr reader.
    """

    def __init__(self, script: str) -> None:
        super().__init__(
            binary="/bin/sh",
            model="/models/network.bin.gz",
            config="/worker/configs/analysis-gpu.cfg",
        )
        self._script = script

    def command(self) -> list[str]:
        return ["/bin/sh", "-c", self._script]


class TestAnEngineThatRefusesToStart:
    async def test_it_reports_what_katago_said(self) -> None:
        """The whole point of keeping the tail.

        Dying at startup is the ordinary way this fails — a bad config key, a
        model it cannot read, a card it cannot find — and KataGo explains
        itself on stderr and nowhere else. A worker on RunPod has no shell to
        go and reproduce it in, so an error that omits this is an error nobody
        can act on.
        """
        broken = ScriptedEngine("echo \"Unused key 'logDir' in config\" >&2; exit 1")

        with pytest.raises(EngineNotRunning) as caught:
            await broken.start()

        assert "Unused key 'logDir'" in str(caught.value)
        assert "code 1" in str(caught.value)

    async def test_it_does_not_sit_out_the_timeout(self) -> None:
        """Waiting three minutes for a process that exited two seconds ago is
        the least useful way to find out that it exited."""
        broken = ScriptedEngine('echo "boom" >&2; exit 3')

        started = time.monotonic()

        with pytest.raises(EngineNotRunning):
            await broken.start()

        assert time.monotonic() - started < 10

    async def test_it_says_so_when_katago_dies_silently(self) -> None:
        broken = ScriptedEngine("exit 127")

        with pytest.raises(EngineNotRunning) as caught:
            await broken.start()

        assert "wrote nothing to stderr" in str(caught.value)

    async def test_it_keeps_only_the_tail(self) -> None:
        """KataGo prints its whole configuration at startup. The last lines are
        the ones that say why it stopped."""
        broken = ScriptedEngine(
            'for i in $(seq 1 200); do echo "line $i" >&2; done; exit 1'
        )

        with pytest.raises(EngineNotRunning) as caught:
            await broken.start()

        message = str(caught.value)

        assert "line 200" in message
        assert "line 1\n" not in message
