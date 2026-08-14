# katasensei-worker

Runs [katasensei](https://github.com/veerasaroot/katasensei) — a KataGo fork
with a teaching-report engine — and feeds its output back to siamgoplay.

Deployed as a **RunPod serverless endpoint**. There is no box: RunPod starts a
container on a GPU when there is work and stops it when there is not, and
siamgoplay pays for the seconds in between.

## What has and has not been verified

The protocol handling is tested against recorded engine output: which message
belongs to which query, when a query is finished, what to do with messages that
arrive out of order, how the query is built, and how the serverless handler
dispatches. The Python suite runs in under a second and needs nothing installed
beyond `requirements-dev.txt`.

**KataGo itself has still not been built or run here.** That needs a C++
toolchain, a CUDA installation and several hundred megabytes of neural net.
Everything below about GPU throughput is an expectation to measure on the card
you actually rent, not a measurement — the first thing to do on a real endpoint
is the sanity checks further down.

## Why Python

The engine is a long-lived child process whose line-based stdout has to be
multiplexed: several analysis threads answer in parallel, and teaching reports
arrive on a separate unordered stream that must be joined on `(id, turnNumber)`.
asyncio does that in a few dozen lines. PHP-FPM cannot hold a child process at
all, and a PHP CLI daemon doing it is markedly more fragile for no gain.

## Two entry points, one engine

| | |
|---|---|
| `app/handler.py` | The RunPod serverless handler. **Production.** |
| `app/main.py` | The same engine behind FastAPI. Development, or a box you own. |

Both take the request shapes in `app/schemas.py`, so what runs on RunPod cannot
accept something the development server refuses.

The difference that matters is **where the queue lives**. `main.py` keeps its
own: hand it a review and it answers "queued" at once, because the process will
still be there in eight minutes. On RunPod it will not be — the platform hands a
worker one job, waits for the handler to return and is free to stop the
container the moment it does. So `handler.py` runs a review to completion inside
the call, and RunPod's queue is the queue.

What does *not* change is where results go. Reports are pushed to Laravel over
the signed callback as the engine produces them, not returned from the handler:
a job's output is capped well below a game's worth of teaching reports, and the
player is watching a progress bar that has to move while the review runs.

## Cold starts

The endpoint holds no machine when nothing is happening. That is the point — a
Go server for one country is quiet for most of the day, and a GPU rented by the
hour to sit idle costs more in a month than the service earns.

The bill for it is the first request after a quiet spell: pull the image, claim
a card, load the neural net. Tens of seconds. Two things soften it.

**The engine outlives the job.** `handler.py` starts KataGo on the first job and
keeps it for the container's life, so only the first request to a given worker
pays the load. Every one after it answers in the time the search takes.

**The idle timeout is set long.** A player on the analysis board clicks through
a game over several minutes; at 300 seconds of idle timeout they pay one cold
start for the session rather than one per position.

The analysis board says which wait it is on, because a spinner that means one
second and a spinner that means fifty are otherwise the same spinner.

## Endpoint settings

| Setting | Value | Why |
|---|---|---|
| Max workers | 3 | One is enough for the load; the rest are so a long review does not sit in front of somebody's analysis board. |
| Idle timeout | 300s | Long enough that a play session pays one cold start. |
| Execution timeout | 900s | Above the slowest plausible review, below "a wedged engine bills all night". |
| FlashBoot | on | Cuts the cold start when the endpoint has had recent traffic. |
| GPU | 24GB class (4090 / L4 / A5000) | The human-rank net is 18 blocks. |

One endpoint serves both reviews and the analysis board. A second one would
double the number of workers going cold and halve the chance a player's first
request finds one already awake.

## Building it

```bash
docker build -t katasensei-worker .
```

The image is CUDA-only. `./katago runtests` runs during the build — it covers
the board, the rules and the report invariants and touches no neural net, so a
broken build is caught on a GPU-less builder rather than on somebody's first
review.

Models can be baked in, which is the better default on serverless: RunPod caches
the image on the worker, so a net that is part of it is already local when a
cold start begins, while a network volume is a read over the wire at exactly the
moment somebody is waiting — and pins the endpoint to one datacentre.

```bash
docker build -t katasensei-worker \
  --build-arg KATAGO_MODEL_URL=https://.../kata1-b18c384nbt.bin.gz \
  --build-arg KATAGO_HUMAN_MODEL_URL=https://.../b18c384nbt-humanv0.bin.gz .
```

Leave both out and the image expects `/models` instead — a network volume, or
`-v` on a machine you own.

For development without Docker:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
```

```bash
.venv/bin/python -m pytest
```

## Environment

| | |
|---|---|
| `CALLBACK_SECRET` | Signs result batches. Must match Laravel's `KATAGO_CALLBACK_SECRET`. |
| `LARAVEL_URL` | Now a public HTTPS address — see below. |
| `KATAGO_MODEL`, `KATAGO_CONFIG` | Set by the image; override to point at a volume. |
| `KATAGO_HUMAN_MODEL` | Unset disables rank comparison for every tier. |
| `ANALYSIS_TIMEOUT` | How long one position may take before a wedged engine is abandoned. Default 60s. |
| `API_TOKEN` | **FastAPI only.** Empty means that face refuses everything. RunPod's own key is the lock on the endpoint. |

Leaving `KATAGO_HUMAN_MODEL` unset makes reports come back with
`humanComparison: null` and `degraded: ["NO_HUMAN_MODEL"]`, which is exactly
what the free tier is meant to produce — not a misconfiguration.

## The callback is on the open internet now

There is no private network any more. RunPod workers come up on whatever address
the platform gives them and go away again, so there is nothing to allow and
nothing to sit behind: **the signature is the whole of the lock**, where it used
to be the second of two.

Which is why it covers a timestamp as well as the body. A signature over the
body alone is valid forever, and the final batch of a review is precisely the
message that marks it finished — capture one and it could be replayed at any
point. Laravel refuses anything more than five minutes out of step.

## Models

| File | What it is | Needed for |
|---|---|---|
| `kata1-b18c384nbt-*.bin.gz` | 18-block net | The main net. Worth running now that a GPU is doing the work. |
| `kata1-b15c192-*.bin.gz` | 15-block net | Lighter, if the card is small. |
| `b18c384nbt-humanv0.bin.gz` | Human-rank net | The "what would a 5 kyu play here" comparison. |
| `configs/analysis-gpu.cfg` | Engine config | In this repo. Diff it against `cpp/configs/analysis_example.cfg` in the version you build. |

The human net was the single most expensive thing in a CPU review, and it is why
the tiers differ in whether it is loaded at all. On a GPU it is much cheaper — if
the plans are ever reshaped, that is the assumption to revisit.

## How fast this is

Expect roughly **1000–2000 playouts per second** on a 24GB card with the main
net, against 10–20 on the CPU backend this used to run on. A 200-move game at
600 visits is 120,000 visits, so on the order of **one to two minutes** rather
than eight to twelve.

**Measure it before believing it.** The number depends on the card, the net and
whether FP16 checks out, and none of it has been run here.

A review is still a queued job with a notification. Two minutes is not a button
you hold a page open for, and the cold start alone can exceed the search.

Reviews still run one at a time per worker. The engine already parallelises
across its analysis threads, and RunPod hands a worker one job anyway.

## Sanity checks on a new endpoint

`docs/TeachingReport.md` in katasensei names four, and they are worth running on
the first output from any new configuration:

- `quality.residualVsLeadDelta` below about 1.0 on most moves — the ownership
  change really does account for the score change.
- `moveEval.pointsLostDisagreement` below about 1 point on most moves.
- `humanComparison.moverRankExpectation.coverage` above 0.7. If it is routinely
  lower, raise `humanSLRootExploreProbWeightless` or `maxVisits`; below that the
  rank comparison is unreliable and the fact sheet will withhold it.
- Reviewing the same game with the colours swapped gives identical `pointsLost`.

And before any of those, in a shell on the endpoint:

```bash
katago runtests
```

which exercises the sensei invariants and needs no neural net at all.

## What a job looks like

One endpoint, so `kind` says which of the two it is.

```json
{"input": {"kind": "analyze", "query_id": "live-1", "moves": [["B", "Q16"]],
           "board_x_size": 19, "max_visits": 400}}
```

```json
{"input": {"kind": "review", "review_id": "01J...", "moves": [["B", "Q16"]],
           "max_visits": 600, "student_rank": "5k", "use_human_model": true}}
```

`analyze` is answered in the job's output. `review` returns a snapshot, and the
reports themselves arrive at
`POST /api/internal/reviews/{id}/reports` as the engine produces them.

Winrate and score come back from **Black's** point of view, converted once in
`summarise()`. KataGo reports them for whoever is to move, which is right for an
engine and wrong for a graph a person reads — the bar would jump to the other
side of the screen on every move while nothing had changed.

## Running it as a plain server

For a laptop, or a machine you own:

```bash
docker run --rm -p 8100:8100 --gpus all \
  -v /srv/models:/models:ro \
  -e CALLBACK_SECRET=... -e API_TOKEN=... \
  -e LARAVEL_URL=http://10.0.0.2 \
  -e KATAGO_HUMAN_MODEL=/models/human.bin.gz \
  katasensei-worker \
  uvicorn app.main:app --host 0.0.0.0 --port 8100
```

Point Laravel at it with `KATAGO_DRIVER=http`.

| | |
|---|---|
| `GET /healthz` | Whether the engine is up and how deep the queue is. |
| `POST /v1/review` | Start a review. Returns immediately; results are pushed back. |
| `GET /v1/review/{id}` | Progress, for the rare case Laravel needs to ask. |
| `POST /v1/analyze` | One position, answered on the same connection. |

Everything but `/healthz` needs `Authorization: Bearer $API_TOKEN`, and an empty
`API_TOKEN` closes that face entirely rather than opening it.

`POST /v1/review` is idempotent on `review_id`: a retry gets the existing job
rather than starting a second review of the same game.
