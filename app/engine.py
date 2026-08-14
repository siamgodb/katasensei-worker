"""The KataGo child process.

One `katago analysis` process is started when the worker boots and kept for its
lifetime: loading a neural net takes several seconds, which would otherwise be
paid on every review. Queries go in on stdin, answers come back on stdout in
whatever order the analysis threads finish them, and a reader task fans them out
to whoever is waiting.

This is why the worker is Python rather than PHP. Holding a child process open
and multiplexing its line-based output is what asyncio is for; PHP-FPM cannot do
it at all, and a PHP CLI daemon doing it is a great deal more fragile.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from collections import deque
from typing import Any, Callable

from .protocol import EngineError, ProtocolError, Response, parse_line

log = logging.getLogger(__name__)

STDERR_TAIL_LINES = 60
"""How much of KataGo's own complaining to keep.

It writes its configuration and per-search timings here, which is noise until
the moment it refuses to start — and then it is the only account of why. On a
serverless worker there is no shell to go and reproduce it in, so whatever is
not kept here is gone.
"""

Listener = Callable[[Response], bool]
"""Takes a response; returns whether it belonged to the listener."""


class EngineNotRunning(RuntimeError):
    pass


class KataGoEngine:
    def __init__(
        self,
        binary: str,
        model: str,
        config: str,
        human_model: str | None = None,
        *,
        extra_args: list[str] | None = None,
    ) -> None:
        self._binary = binary
        self._model = model
        self._config = config
        self._human_model = human_model
        self._extra_args = extra_args or []

        self._process: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task[None] | None = None
        self._stderr_reader: asyncio.Task[None] | None = None
        self._listeners: list[Listener] = []
        self._write_lock = asyncio.Lock()
        self._ready = asyncio.Event()
        self._stderr_tail: deque[str] = deque(maxlen=STDERR_TAIL_LINES)
        self._died = asyncio.Event()
        self._stderr_level = (
            logging.INFO if os.environ.get("KATAGO_LOG_STDERR") == "1" else logging.DEBUG
        )

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    def command(self) -> list[str]:
        argv = [
            self._binary,
            "analysis",
            "-config",
            self._config,
            "-model",
            self._model,
        ]

        if self._human_model:
            argv += ["-human-model", self._human_model]

        return argv + self._extra_args

    async def start(self) -> None:
        if self.is_running:
            return

        argv = self.command()
        log.info("starting engine: %s", " ".join(argv))

        self._stderr_tail.clear()
        self._died.clear()

        self._process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        self._reader = asyncio.create_task(self._read_stdout(), name="katago-stdout")
        self._stderr_reader = asyncio.create_task(self._read_stderr(), name="katago-stderr")

        # KataGo loads its nets before it will answer anything. Rather than
        # sleeping a guessed number of seconds, ask it something trivial and
        # wait for the reply.
        await self._await_ready()

    async def stop(self) -> None:
        for task in (self._reader, self._stderr_reader):
            if task is not None:
                task.cancel()

                with contextlib.suppress(asyncio.CancelledError):
                    await task

        self._reader = self._stderr_reader = None

        if self._process is not None and self._process.returncode is None:
            self._process.terminate()

            try:
                await asyncio.wait_for(self._process.wait(), timeout=10)
            except asyncio.TimeoutError:  # pragma: no cover - only on a wedged engine
                self._process.kill()

        self._process = None
        self._ready.clear()

    async def send(self, query: dict[str, Any]) -> None:
        if not self.is_running or self._process is None or self._process.stdin is None:
            raise EngineNotRunning("the analysis engine is not running")

        line = json.dumps(query, separators=(",", ":")) + "\n"

        # One writer at a time: the engine reads whole lines, and two coroutines
        # interleaving their writes would produce one corrupt query and one
        # silently lost.
        async with self._write_lock:
            self._process.stdin.write(line.encode())
            await self._process.stdin.drain()

    def add_listener(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def remove_listener(self, listener: Listener) -> None:
        with contextlib.suppress(ValueError):
            self._listeners.remove(listener)

    def dispatch(self, response: Response) -> None:
        """Offer a response to each listener until one claims it.

        Exposed so the routing can be tested against recorded engine output
        without a running engine.
        """
        for listener in list(self._listeners):
            try:
                if listener(response):
                    return
            except Exception:  # pragma: no cover - a listener must not kill the reader
                log.exception("listener raised while handling a response")

        # An answer to a query nobody is waiting for: the requester gave up, or
        # the engine is replying to something from before a restart. Worth a log
        # line, not worth an exception.
        log.debug("unclaimed response: %r", response)

    def stderr_tail(self) -> str:
        """The last thing KataGo said before it stopped saying anything."""
        return "\n".join(self._stderr_tail) or "(it wrote nothing to stderr)"

    async def _await_ready(self) -> None:
        probe = {"id": "__ready__", "action": "query_version"}

        def listener(response: Response) -> bool:
            if isinstance(response, EngineError) and response.query_id == "__ready__":
                self._ready.set()

                return True

            return False

        self.add_listener(listener)

        ready = asyncio.ensure_future(self._ready.wait())
        died = asyncio.ensure_future(self._died.wait())

        try:
            # A process that has already exited makes this fail, and the
            # reason it exited is worth far more than "could not write to it" —
            # so the failure is swallowed and the wait below reports instead.
            with contextlib.suppress(Exception):
                await self.send(probe)

            # Whichever happens first. Waiting only for readiness would sit out
            # the full timeout after a process that has already exited, which
            # is the most common way this fails and the least useful way to
            # learn about it.
            done, _ = await asyncio.wait(
                {ready, died},
                timeout=180,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if ready in done:
                return

            raise EngineNotRunning(
                f"the analysis engine exited before it was ready "
                f"(code {self._process.returncode if self._process else '?'}). "
                f"Its last words:\n{self.stderr_tail()}"
                if died in done
                else f"the analysis engine did not become ready within 180s. "
                f"Its last words:\n{self.stderr_tail()}"
            )
        finally:
            for task in (ready, died):
                task.cancel()

            self.remove_listener(listener)

    async def _read_stdout(self) -> None:
        assert self._process is not None and self._process.stdout is not None

        while True:
            raw = await self._process.stdout.readline()

            if not raw:
                # The engine is gone. This line used to be the whole account of
                # it, which told you nothing you could act on — KataGo explains
                # itself on stderr, so that is what gets printed.
                self._died.set()
                log.error(
                    "engine stdout closed; katago said:\n%s",
                    self.stderr_tail(),
                )
                break

            try:
                response = parse_line(raw.decode(errors="replace"))
            except ProtocolError:
                log.exception("could not parse a line of engine output")
                continue

            if response is None:
                continue

            # A version reply is not an error, but it is also not a response
            # shape this protocol models; treat it as the readiness signal.
            if self._probe_reply(raw):
                self._ready.set()
                continue

            self.dispatch(response)

    @staticmethod
    def _probe_reply(raw: bytes) -> bool:
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            return False

        return isinstance(message, dict) and message.get("id") == "__ready__"

    async def _read_stderr(self) -> None:
        assert self._process is not None and self._process.stderr is not None

        while True:
            raw = await self._process.stderr.readline()

            if not raw:
                break

            line = raw.decode(errors="replace").rstrip()

            # Kept whether or not anybody is listening, because the moment it
            # matters is the moment the process is already gone.
            self._stderr_tail.append(line)

            # KataGo logs its configuration and per-search timings here. Noise
            # during a normal run; set KATAGO_LOG_STDERR=1 to watch it live.
            log.log(self._stderr_level, "katago: %s", line)
