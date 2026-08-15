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


class TestTheReadinessHandshake:
    """The one exchange that had never been run against a real engine.

    KataGo answers the probe with a version string. Nothing in this protocol
    models that shape, so `parse_line` returns None for it — and the guard that
    drops None used to sit in front of the check that recognises it, which made
    every start wait out the full timeout on an engine that had been ready for
    three minutes. Every other test used a fake engine and never sent a line
    down a real pipe.
    """

    async def test_a_version_reply_means_ready(self) -> None:
        katago = ScriptedEngine(
            'read -r line; echo \'{"id":"__ready__","version":"1.17.2"}\'; exec sleep 30'
        )

        await katago.start()

        assert katago.is_running

        await katago.stop()

    async def test_an_error_reply_also_means_ready(self) -> None:
        """A version of KataGo that does not know the action still proves it is
        listening, which is the only thing being asked."""
        katago = ScriptedEngine(
            'read -r line; echo \'{"id":"__ready__","error":"unknown action"}\'; exec sleep 30'
        )

        await katago.start()

        assert katago.is_running

        await katago.stop()

    async def test_the_probe_reply_is_not_offered_to_listeners(self) -> None:
        seen: list = []

        katago = ScriptedEngine(
            'read -r line; echo \'{"id":"__ready__","version":"1.17.2"}\'; exec sleep 30'
        )
        katago.add_listener(lambda response: bool(seen.append(response)) or True)

        await katago.start()

        assert seen == []

        await katago.stop()


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
