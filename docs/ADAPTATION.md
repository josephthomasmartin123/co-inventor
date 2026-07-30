# Adapting Co-Scientist to invention

Co-Inventor is an implementation of Google's AI Co-Scientist, pointed at a different target.
The reference throughout is *Accelerating scientific discovery with Co-Scientist* (Gottweis,
Weng, Daryin, Tu et al., Google Cloud AI Research / Google DeepMind / Google Research);
local copy in the repository root as `google coscientist paper.pdf`. Quotations below are
from that PDF — extract it with `pdftotext -layout` to grep for context.

Co-Scientist produces
**scientific hypotheses**; Co-Inventor produces **patentable inventions**. The six agents,
the Elo tournament and the evolution loop are the paper's. What changes is what counts as a
good output — and that changes more than it first appears.

This document records every place we deliberately depart from the paper, and why. It is
organised into three parts:

1. **Domain adaptations** — changes required because an invention is not a hypothesis.
2. **Architectural simplifications** — places we implement less than the paper.
3. **Known gaps** — paper features not implemented, listed so they aren't mistaken for
   decisions.

The distinction matters. Anything in part 1 should be defended on invention grounds if
challenged. Anything in parts 2 and 3 is a candidate for future work, not a design position.

---

## 1. Domain adaptations

### 1.1 A hypothesis is testable; an invention must be claimable

The paper's outputs are hypotheses and research plans, judged partly on **testability** —
whether an experiment could confirm them. That criterion doesn't transfer. An invention is
judged on whether it can be *built* and *claimed*, which is why the generation prompt
refuses outcomes and demands mechanisms:

> WEAK: "Improved thermal management for LEDs"
> STRONG: "Micro-channel heat spreader with phase-change fluid embedded in the LED
> substrate; capillary wicking passively distributes and evaporates coolant"

**Where:** `app/agents/generation.py` — `_BASE_SYSTEM`.

### 1.2 Review dimensions: correctness → patentability

The paper's Reflection agent verifies the "correctness, quality, and novelty" of generated
outputs, with a preliminary assessment of safety and ethics in its initial review, against
system-level criteria including "plausibility, novelty, testability, and safety" and
alignment with the research goal. We keep novelty and plausibility, and replace the rest:

| Paper | Co-Inventor | Weight | Why |
| --- | --- | --- | --- |
| Novelty | Novelty | 0.30 | Retained, but re-based on prior art — see 1.3 |
| Correctness / quality | Scientific plausibility | 0.25 | Does the physics or chemistry actually work |
| Testability | Feasibility | 0.15 | Buildable with current or near-future technology, not confirmable by experiment |
| — | **Patentability** | 0.20 | New. A specific, claimable technical mechanism rather than an abstract idea or a desired result |
| Alignment with research goal | Problem fit | 0.10 | Addresses the root cause, not a symptom |
| Safety / ethics | *not scored* | — | See 3.4 |

Scores are combined into a weighted `overall_score`.

**Where:** `app/agents/reflection.py` — `WEIGHTS`, `_FULL_SYSTEM`.

### 1.3 Novelty means prior art, and absence of prior art is not enough

For a hypothesis, novelty is roughly "not already published". For an invention it is a
two-part test: absent from prior art **and** non-obvious. The second part has no analogue
in the paper, and it is the part that most often fails.

So the review searches patents and prior art (capturing conflicting URLs on the `Review`),
and the novelty criterion carries an explicit instruction that a finding of "no exact match"
does not earn a high score when the arrangement is an obvious collocation of known elements.

**Where:** `app/agents/reflection.py` — novelty and patentability criteria in `_FULL_SYSTEM`,
`Review.prior_art_found` in `app/models/invention.py`.

### 1.4 The aggregation rule: combination requires a new technical effect

**This is the largest single adaptation.** The paper mentions it once, in its out-of-the-box
generation prompt — *"This should not be a mere aggregation of existing methods or
entities"* — and combines freely elsewhere, including a Combination strategy that
"attempts to directly combine the best aspects of several top-ranking hypotheses".

For hypotheses, that is harmless: two ideas stated together are just a richer idea. For
patent claims it is fatal. Features that do not functionally interact are an *aggregation* —
EPO's aggregation-versus-combination line, and the "predictable combination of known
elements" bar from *KSR v. Teleflex*. Each element keeps doing its own job, and an examiner
reads the claim as obvious. More features is not more invention.

So the rule is enforced structurally, in five places rather than one:

| Where | Enforcement |
| --- | --- |
| `evolution.py` · `_combine_prompt` | A four-step interaction test; must name the interaction and the effect; may return `combination_viable: false` |
| `evolution.py` · `_evolve_one` | Honours that refusal — the candidate is dropped, with the reason logged |
| `evolution.py` · `SYSTEM_PROMPT` | Additions must change what the mechanism *does*, not what it *contains* |
| `generation.py` · debate synthesis | Experts' mechanisms must interact; otherwise deepen one instead of bundling all three |
| `reflection.py` · both tiers | Aggregation test on novelty and patentability; non-interacting claim features cap patentability at 1–2 |
| `ranking.py` · debate criteria | Aggregation is penalised: one mechanism with a clear effect beats a longer feature list |

**Zero combinations is a valid outcome.** The system is expected to decline rather than
manufacture a hybrid to fill a slot.

### 1.5 Combination pairs on mechanistic family, not rank

The paper combines "several top-ranking hypotheses" and gives no pairing rule. Rank is the
wrong basis: it measures how good each invention is *alone*, which says nothing about
whether two mechanisms interact. Worse, the top two are often near-variants of one
approach — the pairing least likely to yield anything.

Co-Inventor therefore reuses the proximity clusters (which the pipeline already computes and
previously discarded) to pair across **different** mechanistic families, capping attempts at
`MAX_COMBINATION_ATTEMPTS = 2` and using rank only to break ties between eligible pairs. If
every top-ranked pair sits in one family, no combination is attempted at all.

Note the symmetry with the paper's *ranking* use of proximity, which is the opposite and
also correct: **compare** similar ideas, because that is the informative match; **combine**
dissimilar ones, because that is where a new effect can come from.

**Where:** `app/agents/evolution.py` — `_combination_candidates`.

### 1.6 Invention triggers

New, with no counterpart in the paper. The `literature_exploration` strategy records the
specific recent advance that makes an invention newly conceivable:

- `trigger_advance` — what the advance is
- `trigger_source_domain` — the field it came from
- `trigger_url` — the paper, article or patent it was found in

The motivating case is a real one: nanobubble research from **food science** triggering an
invention for concrete workability. For a patent this provenance is evidence, not
decoration — it supports the non-obviousness argument and dates the idea's availability.
Evolved variants inherit their parent's trigger so the provenance survives refinement.

**Where:** `app/models/invention.py`, `app/agents/generation.py` —
`_OUTPUT_SCHEMA_LITERATURE`.

### 1.7 Five generation strategies instead of four

The paper lists literature exploration, simulated scientific debate, iterative assumptions
identification, and research expansion. We split research expansion in two, because the two
halves behave very differently for invention:

- `direct` — first-principles decomposition to the limiting physical phenomenon
- `analogical` — the same challenge located in unrelated domains, whose solved mechanism is
  then adapted, naming the source domain explicitly

Cross-domain analogy is promoted to a first-class strategy because distant-domain transfer
is the most reliable route to a non-obvious mechanism. Each strategy returns two concepts.

**Where:** `app/agents/generation.py`.

### 1.8 Tournament criteria include commercial defensibility

The paper's debates weigh novelty, correctness and testability. Ours weigh novelty,
mechanism specificity, **how hard the invention would be for a competitor to design
around**, feasibility and problem fit. Design-around difficulty is a patent-value question
with no scientific analogue.

**Where:** `app/agents/ranking.py` — `SYSTEM_PROMPT`.

### 1.9 Meta-review output is shaped for an inventor

Same role as the paper's research overview, different fields: `strongest_approaches`,
`recurring_challenges`, `unexplored_directions`, `cross_domain_insight`, `recommendation`.
The cross-domain field exists for the same reason as 1.7.

**Where:** `app/models/pipeline.py` — `MetaReviewResult`, `app/agents/meta_review.py`.

---

## 2. Architectural simplifications

These are not domain adaptations. They are places where we implement the paper's design in a
smaller form, and each one has a consequence worth knowing.

### 2.1 A fixed six-stage sequence instead of a Supervisor agent

The paper runs a **Supervisor agent** over an asynchronous task queue with a worker pool and
a context memory, periodically computing statistics and "strategically weighting and
sampling the specialized agents for execution via the worker processes". Generation, review, ranking and evolution
interleave continuously.

Co-Inventor runs the six stages once, in order, per session. This is far easier to follow
and to stream to a UI, and it is why the paper's continuous competition had to be
implemented as an explicit second pass — see 2.2.

**Consequence:** compute cannot be reallocated mid-run. A promising direction cannot be
given more generation budget while it is still cheap to explore.

### 2.2 Tournament re-entry as a second pass

The paper's evolution is safe to run speculatively because of one rule:

> The Evolution agent generates new hypotheses; it doesn't modify or replace existing ones.
> This strategy protects the quality of top-ranked hypotheses from flawed improvements, as
> each new hypothesis must also compete in the tournament.

Under a Supervisor this needs no special handling — a new hypothesis is simply another
competitor. In a fixed sequence it does. Stage 5 therefore runs: evolve → review the
variants → re-run the tournament over the merged field, with the variants' ids passed as
`new_ids` so match prioritisation actually tests them.

Two invariants hold this together and should not be broken:

- **Nothing gets a rating advantage for its origin.** `Invention.elo_score` defaults to
  1200 — the paper's initial rating for any newly added hypothesis — and only match
  outcomes move it. An earlier implementation seeded variants at `parent_elo + 50`, which
  quietly promoted them into the final five without evaluation.
- **`ranking.run` is safe to call more than once.** It no longer seeds ratings from review
  scores, so a second pass adds newcomers without discarding what the first pass
  established. Reviews reach the judge inside the debate prompt, where they inform the
  comparison without pre-deciding the order.

**Where:** `app/orchestrator.py` stage 5b, `app/agents/ranking.py`.

### 2.3 Cluster labels instead of a proximity graph

The paper's Proximity agent "asynchronously computes a proximity graph". We ask the model
for cluster assignments — a label per invention plus a theme — which is coarser but enough
for the two jobs we need it for: de-duplication, and the family logic in 1.5 and 2.2.

**Where:** `app/agents/proximity.py`.

### 2.4 The rapid review pass scores down, it does not remove

The paper's initial review "aims to quickly discard flawed, non-novel, or otherwise
unsuitable hypotheses". Ours scores a failing concept down and lets it continue into the
tournament, where it loses on merit. Nothing is removed from a run except near-duplicates
(stage 2) and combinations that stage 5 skips or declines.

This is arguably a bug rather than a decision. It is recorded here because the code reads
like a gate and is not one.

**Where:** `app/agents/reflection.py` — `initial_pass`, and `app/orchestrator.py`, which
passes the full list to ranking.

---

## 3. Known gaps

Paper features not implemented. Listed so nobody mistakes their absence for a judgement.

### 3.1 Meta-review feedback propagation

The paper's most interesting mechanism: the Meta-review agent's critique "is simply appended
to their prompts in the next iteration", giving feedback propagation without any
fine-tuning. We generate the overview but never feed it back into agent prompts. The
between-rounds loop passes prior top inventions and the overview into *generation* only, and
only when a human starts another round.

### 3.2 Multi-turn debates for top-ranked pairs

The paper uses multi-turn scientific debates for top-ranked hypotheses and single-turn
comparisons for the rest. All our matches are single-turn. This is the remaining ranking
deviation, and unlike the others it costs additional model calls per match.

### 3.3 Additional Reflection review types

The paper describes deep verification, observation review, simulation review and tournament
review beyond the two tiers we implement.

### 3.4 Safety review

The paper scores safety and ethics, and discusses it at length. We do not score it at all.
Defensible for most industrial invention work and indefensible for some of it; if
Co-Inventor is ever pointed at anything where a mechanism could cause harm, this is the
first gap to close.

### 3.5 Scientist-contributed ideas

The paper lets a scientist add their own hypotheses into the tournament and discuss via
chat. We accept a problem statement and an optional steering note between rounds.

---

## Parameters

| Parameter | Value | Defined in |
| --- | --- | --- |
| Generation strategies | 5, concurrent | `agents/generation.py` |
| Concepts per strategy | 2 | `agents/generation.py` |
| Tournament debates | 12 per pass | `config.py` · `elo_rounds` |
| Entry rating, everything | 1200 | `agents/ranking.py` · `INITIAL_ELO` |
| Elo K-factor | 32 | `agents/ranking.py` |
| Match prioritisation | similar · newer · top-ranked | `agents/ranking.py` · `_pick_pair` |
| Concepts evolved | top 3 | `config.py` · `top_k_for_evolution` |
| Combination attempts | ≤ 2, cross-family only | `agents/evolution.py` · `MAX_COMBINATION_ATTEMPTS` |
| Review weights | ·30 ·25 ·20 ·15 ·10 | `agents/reflection.py` · `WEIGHTS` |
| Reviews in flight | 4 | `agents/reflection.py` |
| Search providers | Exa → Tavily → DuckDuckGo | `tools/web_search.py` |

A cross-check implementation of the paper, useful when its prose is ambiguous:
[conradry/open-coscientist-agents](https://github.com/conradry/open-coscientist-agents) —
`coscientist/ranking_agent.py`, `evolution_agent.py` and `framework.py`. It independently
uses an entry rating of 1200 and routes evolved hypotheses through review before the
tournament.
