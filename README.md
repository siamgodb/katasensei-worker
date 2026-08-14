# katasensei-worker

Runs [katasensei](https://github.com/veerasaroot/katasensei) — a KataGo fork
with a teaching-report engine — and feeds its output back to siamgoplay.

Sits on its own box. Laravel asks for a review over a private network and the
worker pushes results back as they are produced; nothing here is reachable from
the internet.

## What has and has not been verified

The protocol handling is tested against recorded engine output: which message
belongs to which query, when a query is finished, what to do with messages that
arrive out of order, and how the query is built.

**KataGo itself has not been built or run here.** That needs a C++ toolchain
and several hundred megabytes of neural net. The first thing to do on a real
box is the sanity checks below — they are what would catch a wrong config,
which the unit tests cannot.

## Why Python

The engine is a long-lived child process whose line-based stdout has to be
multiplexed: several analysis threads answer in parallel, and teaching reports
arrive on a separate unordered stream that must be joined on `(id, turnNumber)`.
asyncio does that in a few dozen lines. PHP-FPM cannot hold a child process at
all, and a PHP CLI daemon doing it is markedly more fragile for no gain.

## Running it

```bash
docker build -t katasensei-worker .
```

```bash
docker run --rm -p 8100:8100 \
  -v /srv/models:/models:ro \
  -e CALLBACK_SECRET=... -e API_TOKEN=... \
  -e LARAVEL_URL=http://10.0.0.2 \
  -e KATAGO_HUMAN_MODEL=/models/b18c384nbt-humanv0.bin.gz \
  katasensei-worker
```

For development without Docker:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
```

```bash
.venv/bin/python -m pytest
```

## Models

| File | What it is | Needed for |
|---|---|---|
| `kata1-b10c128-*.bin.gz` | 10-block net | The main net. Fast enough to review on CPU. |
| `kata1-b15c192-*.bin.gz` | 15-block net | Stronger, for the Pro tier or a GPU box. |
| `b18c384nbt-humanv0.bin.gz` | Human-rank net | The "what would a 5 kyu play here" comparison. |
| `analysis.cfg` | Engine config | Start from `cpp/configs/analysis_example.cfg`. |

Leave `KATAGO_HUMAN_MODEL` unset to run without rank comparison. Reports then
come back with `humanComparison: null` and `degraded: ["NO_HUMAN_MODEL"]`, which
is exactly what the free tier is meant to produce — not a misconfiguration.

The human net is an 18-block model and is the single most expensive thing in a
CPU review. It is the reason the tiers differ in whether it is loaded at all.

## How slow this actually is

KataGo's own README puts the Eigen (CPU) backend at **10–20 playouts per second**
on a good CPU with a 15- or 20-block net. A 200-move game at 100 visits per
position is 20,000 visits, so roughly **8–12 minutes**.

That is why a review is a queued job with a notification and not a button that
waits. Anything in the app that assumes otherwise is wrong about the hardware,
not about the code.

Reviews run one at a time. The engine already parallelises across its analysis
threads; running two games at once makes both finish late and makes the
progress bar lie about which.

## Sanity checks on a new box

`docs/TeachingReport.md` in katasensei names four, and they are worth running
on the first output from any new configuration:

- `quality.residualVsLeadDelta` below about 1.0 on most moves — the ownership
  change really does account for the score change.
- `moveEval.pointsLostDisagreement` below about 1 point on most moves.
- `humanComparison.moverRankExpectation.coverage` above 0.7. If it is routinely
  lower, raise `humanSLRootExploreProbWeightless` or `maxVisits`; below that the
  rank comparison is unreliable and the fact sheet will withhold it.
- Reviewing the same game with the colours swapped gives identical `pointsLost`.

And before any of those:

```bash
katago runtests
```

which exercises the sensei invariants and needs no neural net at all.

## Endpoints

| | |
|---|---|
| `GET /healthz` | Whether the engine is up and how deep the queue is. |
| `POST /v1/review` | Start a review. Returns immediately; results are pushed back. |
| `GET /v1/review/{id}` | Progress, for the rare case Laravel needs to ask. |
| `POST /v1/analyze` | One position, answered on the same connection. |

Everything but `/healthz` needs `Authorization: Bearer $API_TOKEN`. Result
batches are signed with `CALLBACK_SECRET` over the exact request body.

`POST /v1/review` is idempotent on `review_id`: a retry gets the existing job
rather than starting a second review of the same game.

`POST /v1/analyze` is the analysis board, and is the one endpoint that answers
synchronously — somebody is watching the screen, and a callback for something
that takes seconds is two more moving parts to get wrong. It skips the review
queue and goes straight to the engine, which parallelises across its own
analysis threads: a review running at the time finishes slightly later, and the
board gets an answer in seconds rather than in eight minutes. `ANALYSIS_TIMEOUT`
(default 60s) bounds how long the connection is held.

Winrate and score come back from **Black's** point of view, converted once in
`summarise()`. KataGo reports them for whoever is to move, which is right for an
engine and wrong for a graph a person reads — the bar would jump to the other
side of the screen on every move while nothing had changed.
