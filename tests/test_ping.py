"""Asking a worker about itself, before asking it to do anything.

Every failure this endpoint has had in production was a configuration one — a
build that did not land, a model that was not where the environment said, a
`LARAVEL_URL` still pointing at localhost — and each was found by running a
real job on a rented card and reading the traceback. A ping is the same
question for the price of a container start.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app import handler as rp
from app.version import VERSION, fingerprint


@pytest.fixture(autouse=True)
def no_ambient_configuration(monkeypatch):
    """Nothing from the developer's own shell, and no outbound calls."""
    for name in rp.WATCHED_ENV:
        monkeypatch.delenv(name, raising=False)


async def test_it_answers_without_starting_the_engine(monkeypatch):
    """The case worth asking about is the misconfigured worker, and a worker
    whose configuration is wrong is exactly the one that cannot start."""
    monkeypatch.setattr(rp, "_runtime", None)
    monkeypatch.setattr(
        rp.Settings,
        "from_env",
        staticmethod(lambda: pytest.fail("ping read the settings")),
    )

    report = await rp.handler({"id": "rp-ping", "input": {"kind": "ping"}})

    assert report["version"] == VERSION
    assert report["warm"] is False


async def test_the_fingerprint_is_the_one_this_checkout_computes():
    """The whole trick. Compare `python -m app.version` here with what the
    endpoint reports: the same string means the image is this code."""
    report = await rp.ping()

    assert report["fingerprint"] == fingerprint()


async def test_the_fingerprint_changes_when_the_engine_config_changes(tmp_path, monkeypatch):
    """A review is shaped by analysis-gpu.cfg as surely as by query.py, so a
    config edit that did not reach the image has to be visible too."""
    before = fingerprint()

    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "analysis-gpu.cfg").write_text("numAnalysisThreads = 1\n")

    monkeypatch.setattr("app.version._APP", Path(__file__).resolve().parent.parent / "app")
    monkeypatch.setattr("app.version._CONFIGS", configs)

    assert fingerprint() != before


async def test_the_callback_proof_is_the_value_laravel_computes(monkeypatch):
    """The other half is in PingKatagoTest, asserting the same literal.

    Both sides HMAC the same public constant with their own copy of the signing
    key, and compare. Equal means the keys are equal. Written out rather than
    recomputed here, because a test that recomputes the thing it is testing
    would agree with any implementation, including a wrong one.
    """
    from app.callbacks import proof

    assert proof("shared-secret") == "68e47251fc22b68a"


async def test_it_never_reports_the_value_of_a_secret(monkeypatch):
    """This report travels back through RunPod's API and into whatever logs it
    passes on the way. One of these is the callback signing key."""
    monkeypatch.setenv("CALLBACK_SECRET", "hunter2")

    report = await rp.ping()

    assert report["env"]["CALLBACK_SECRET"] is True
    assert "hunter2" not in str(report)


async def test_it_says_which_files_are_missing(monkeypatch, tmp_path):
    model = tmp_path / "network.bin.gz"
    model.write_bytes(b"x" * 1024)

    monkeypatch.setenv("KATAGO_MODEL", str(model))
    monkeypatch.setenv("KATAGO_CONFIG", str(tmp_path / "nope.cfg"))

    report = await rp.ping()

    assert report["files"]["model"] == {
        "set": True,
        "path": str(model),
        "exists": True,
        # Size as well as existence: a model truncated by a build that ran out
        # of disk exists and still will not load.
        "bytes": 1024,
    }
    assert report["files"]["config"]["exists"] is False
    assert report["files"]["human_model"] == {"set": False}


async def test_it_names_a_laravel_url_that_points_at_the_worker_itself(monkeypatch):
    """The default. It looks entirely correct on the endpoint's settings page,
    and it means every finished review is delivered to the container that just
    produced it — minutes of a card, thrown away, reported as success."""
    monkeypatch.setenv("LARAVEL_URL", "http://127.0.0.1:8000")

    async def unreachable(self, url, **_):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(rp.httpx.AsyncClient, "get", unreachable)

    report = await rp.ping()

    assert report["laravel"]["local"] is True
    assert "connection refused" in report["laravel"]["error"]


async def test_an_unset_laravel_url_is_not_dialled(monkeypatch):
    monkeypatch.setattr(
        rp.httpx.AsyncClient,
        "get",
        lambda *_, **__: pytest.fail("ping made a request with nowhere to send it"),
    )

    assert (await rp.ping())["laravel"] == {"configured": False}


async def test_warming_reports_the_reason_the_engine_would_not_start(monkeypatch):
    """Rather than raising. A ping that fails tells you nothing the console
    did not already say; a ping that answers tells you what is wrong."""
    monkeypatch.setattr(rp, "_runtime", None)

    async def refuse() -> None:
        raise RuntimeError("the analysis engine exited before it was ready (code 1)")

    monkeypatch.setattr(rp, "runtime", refuse)

    report = await rp.handler({"id": "rp-ping", "input": {"kind": "ping", "warm": True}})

    assert report["warm"] is False
    assert "exited before it was ready" in report["engine_error"]
