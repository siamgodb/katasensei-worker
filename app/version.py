"""What is actually running on the far end.

A serverless endpoint is built from a git push by a service that cannot be
asked "which commit is this?". The console shows a build, not the code inside
it, and a job that fails on a bug fixed an hour ago is indistinguishable from a
job that fails on a bug still present. That question has already been answered
the expensive way — by firing a real job at a GPU and reading the traceback —
and this is the cheap way to answer it instead.

The fingerprint is a hash of the worker's own source. Run

    python -m app.version

on a laptop and compare it with what a `kind: "ping"` job reports: the same
string means the endpoint is running the code in front of you, and a different
one means the build did not land, whatever the console says.

A hash of the files rather than a git revision on purpose. RunPod builds the
image itself and has nowhere to put a `--build-arg`, and a version constant
somebody has to remember to bump is a version constant that is wrong exactly
when it matters.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

VERSION = "0.3.0"
"""Bumped by hand for humans. The fingerprint is what is actually checked."""

_APP = Path(__file__).resolve().parent
_CONFIGS = _APP.parent / "configs"


def fingerprint() -> str:
    """A short hash over everything that decides how this worker behaves.

    The engine config counts as much as the code — a review is shaped by
    `analysis-gpu.cfg` as surely as by `query.py`, and a config change that did
    not reach the image is the same class of problem.
    """
    digest = hashlib.sha256()

    for path in sorted(_APP.glob("*.py")) + sorted(_CONFIGS.glob("*.cfg")):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())

    return digest.hexdigest()[:12]


if __name__ == "__main__":  # pragma: no cover - a one-line diagnostic
    print(f"{VERSION} {fingerprint()}")
