# Joe-Bot

A multi-agent invention system. You describe a technical problem in plain language; it
generates candidate inventions, searches prior art against each one, ranks them by argument
rather than by score, refines the best, and hands back five with a written overview.

It is an implementation of Google's AI Co-Scientist, re-pointed from **scientific
hypotheses** to **patentable inventions**. That change matters more than it sounds: a
hypothesis is judged on whether it can be tested, an invention on whether it can be built
and claimed. Every deliberate departure from the paper is recorded in
[docs/ADAPTATION.md](docs/ADAPTATION.md).

## How a run works

Six stages, in order. Only generation fans out internally.

| | Stage | What it does |
| --- | --- | --- |
| 1 | **Generation** | Five strategies in parallel — literature exploration, simulated expert debate, inverted assumptions, first principles, cross-domain analogy. Two concepts each. |
| 2 | **Proximity** | Drops near-duplicates, clusters the rest into mechanistic families. |
| 3 | **Reflection** | A rapid pass, then a full review with prior-art search, scored on five weighted dimensions. |
| 4 | **Ranking** | An Elo tournament of pairwise debates. Everything enters at 1200; only match outcomes move a rating. |
| 5 | **Evolution** | Deepens the top three, and combines across mechanistic families where the mechanisms genuinely interact. Variants are then reviewed and must win their rank in a second tournament. |
| 6 | **Meta-review** | Reads every evaluation and writes the overview. |

Two ideas do most of the work:

**Ranked by argument, not by score.** Absolute scores from a reviewer drift and cluster. A
fresh judge asked "which of these two is better, and why" is far more reliable, so position
is earned in head-to-head debates. Nothing is seeded from its review score.

**A combination must produce a new technical effect.** Merging two good ideas into one that
merely contains both is an *aggregation* — each part still doing its own job — and however
impressive it reads, a patent examiner treats it as an obvious collocation of known
elements. So combination is attempted only across unlike approaches, must name the effect
neither parent achieves alone, and is declined outright when it cannot. Zero combinations is
a valid outcome.

## Running it locally

Requires Python 3.11+ and an LLM API key.

```bash
# install (uv)
uv sync
# ...or pip
pip install -r requirements.txt

cp .env.example .env    # then edit — see below
python -m app.main
```

Open http://localhost:8000.

### Configuration

Set in `.env` or the environment. One LLM key is required; everything else has a default.

| Variable | Default | Notes |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | — | Use either this or the Anthropic key. If set, bare model names are prefixed `anthropic/`. |
| `ANTHROPIC_API_KEY` | — | Used when no OpenRouter key is present. |
| `DEFAULT_MODEL` | `claude-sonnet-4-6` | Applies to every agent. |
| `GENERATION_MODEL` … | — | Per-agent overrides: also `REFLECTION_MODEL`, `RANKING_MODEL`, `EVOLUTION_MODEL`, `TRIGGER_MODEL`. Blank falls back to `DEFAULT_MODEL`. |
| `EXA_API_KEY` | — | Search is tried in order: Exa, then Tavily, then a keyless DuckDuckGo fallback. |
| `TAVILY_API_KEY` | — | |
| `ELO_ROUNDS` | `12` | Debates per tournament pass. A run holds two passes. |
| `TOP_K_FOR_EVOLUTION` | `3` | How many top inventions get refined. |
| `DB_PATH` | `co_inventor.db` | SQLite file. |
| `PORT` | `8000` | Railway sets this automatically. |

A run takes roughly 15–25 minutes, dominated by the two tournaments, which debate
sequentially. `ELO_ROUNDS` is the main lever if that is too slow — at the cost of a weaker
tournament for variants to win.

## Layout

```
app/
  main.py          FastAPI app; serves the frontend, retires interrupted sessions on boot
  orchestrator.py  the six stages, in order
  config.py        settings and model resolution
  storage.py       SQLite persistence
  agents/          one module per stage, plus base.py for model calls with tool use
  api/             session endpoints and the SSE progress stream
  models/          Invention, Review, Session, per-stage results
  tools/           web search
frontend/          single-page UI, no build step
docs/ADAPTATION.md every deliberate departure from the paper
```

### API

| | |
| --- | --- |
| `POST /api/sessions` | Start a run. Pass `parent_session_id` and `user_feedback` to run another round seeded with the previous result. |
| `GET /api/sessions` | Recent sessions. |
| `GET /api/sessions/{id}` | Session, inventions, reviews and overview. Also how a client recovers if the live stream drops. |
| `GET /api/sessions/{id}/stream` | Server-sent progress events. Returns immediately if the run has already finished. |

## Deploying

Railway builds from the `Dockerfile` and auto-deploys on every push to `master`; see
[RAILWAY.md](RAILWAY.md) for first-time setup. Attach a Volume and set
`DB_PATH=/data/co_inventor.db`, or sessions are wiped on each redeploy.

> **A deploy kills any run in flight.** A pipeline is an in-process asyncio task, so
> restarting the container ends it. The session is marked failed on the next boot and
> cannot be resumed. Before pushing, check that nothing is mid-run:
>
> ```bash
> curl -s "$APP_URL/api/sessions?limit=5"
> ```
>
> A status of `generating`, `proximity`, `reflecting`, `ranking`, `evolving` or
> `meta_reviewing` means someone is 15–25 minutes into a run. Wait.

The browser side is resilient to everything short of that: the session id lives in the URL,
so a refresh rejoins a run in progress; a dropped stream reconnects; and if it cannot, the
page falls back to polling for the result.
