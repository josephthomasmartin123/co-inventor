"""
Evolution Agent

Takes the top-ranked inventions and produces candidate variants:

1. ENHANCE — Deepen the mechanism: add specificity, address reflection weaknesses,
   strengthen the non-obviousness argument. One enhanced variant per top-K invention.

2. COMBINE — Attempted only between DIFFERENT mechanistic families, and only where the
   two mechanisms interact to produce a technical effect neither parent achieves alone.
   Combining two variants of one approach yields an aggregation, which is worthless as a
   patent claim however impressive it reads. Each attempt may decline, and zero
   combinations is a valid outcome — see docs/ADAPTATION.md §1.4 and §1.5.

This agent PROPOSES; it does not promote. Variants carry no Elo advantage: they enter the
tournament at the same rating as everything else and must win to outrank their parents. The
caller is responsible for reviewing them and re-running the tournament, which is what the
paper requires — "each new hypothesis must also compete in the tournament" (§2.2).
Consequently `EvolutionResult.final_ranked` is the merged field in its PRE-tournament
order; it is not the final ranking despite the name.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable

from app.agents.base import run_claude_with_tools, parse_json_response, AgentCallError
from app.config import settings
from app.models.invention import Invention, Review
from app.models.pipeline import EvolutionResult

logger = logging.getLogger(__name__)

# How many cross-family pairs to attempt combining. Each attempt is a model call, and
# beyond the strongest couple of pairs the returns fall off sharply.
MAX_COMBINATION_ATTEMPTS = 2

SYSTEM_PROMPT = """You are a master inventor and patent strategist.

Your task: take a promising invention concept and make it significantly better.

FIRST PRINCIPLE: an invention is a mechanism that produces a technical effect. Adding
features, steps, or materials that do not interact with the existing mechanism adds
nothing inventive — it is an aggregation, and an examiner will read it as an obvious
collocation of known elements. Every addition you make must change what the mechanism
DOES, not merely what it CONTAINS.

Focus on:
- Making the MECHANISM more specific (add materials, dimensions, process steps, control algorithms)
- Addressing weaknesses identified in the evaluation
- Strengthening non-obviousness — add a secondary effect that arises from the mechanism
  itself, not a bolted-on feature
- Ensuring it is clearly distinguishable from prior art

The output invention should read like a strong patent claim:
specific, technical, and non-obvious.

Output ONLY valid JSON. No prose outside the JSON.

OUTPUT SCHEMA:
{
  "title": "Enhanced invention title",
  "summary": "2-3 sentences: what it is and what makes it better than the base concept",
  "mechanism": "3-4 sentences with concrete specifics — materials, process steps, physics"
}"""


def _enhance_prompt(inv: Invention, review: Review | None, problem: str) -> str:
    weaknesses = []
    if review:
        if review.novelty_score < 4:
            weaknesses.append(f"Novelty ({review.novelty_score}/5): {review.novelty_rationale}")
        if review.feasibility_score < 4:
            weaknesses.append(f"Feasibility ({review.feasibility_score}/5): {review.feasibility_rationale}")
        if review.patentability_score < 4:
            weaknesses.append(f"Patentability ({review.patentability_score}/5): {review.patentability_rationale}")

    weakness_block = (
        "\n".join(f"- {w}" for w in weaknesses)
        if weaknesses else "- No major weaknesses identified — focus on deepening specificity"
    )

    return f"""PROBLEM: {problem}

BASE INVENTION:
Title: {inv.title}
Summary: {inv.summary}
Mechanism: {inv.mechanism}

WEAKNESSES TO ADDRESS:
{weakness_block}

TASK: Produce one significantly improved version of this invention.
Make the mechanism more specific, more novel, or more surprising.
Do not just restate the base invention — make a meaningful advance."""


def _combine_prompt(inv_a: Invention, inv_b: Invention, problem: str) -> str:
    return f"""PROBLEM: {problem}

INVENTION A (ranked #1):
Title: {inv_a.title}
Summary: {inv_a.summary}
Mechanism: {inv_a.mechanism}

INVENTION B (ranked #2):
Title: {inv_b.title}
Summary: {inv_b.summary}
Mechanism: {inv_b.mechanism}

TASK: Decide whether A and B genuinely combine — and only then combine them.

A combination is inventive ONLY if the two mechanisms interact to produce a NEW TECHNICAL
EFFECT that neither achieves alone. Merely putting both features in one device is an
AGGREGATION, not an invention — each part just does its own job, and a patent examiner
will treat it as an obvious collocation of known elements.

  AGGREGATION (worthless): "The panel has A's phase-change coating AND B's micro-channels."
    Each still does exactly what it did separately. Nothing new emerges.
  COMBINATION (inventive): "B's micro-channels wick the fluid that A's coating releases on
    phase change, so the coating self-regenerates — neither mechanism is self-regenerating
    alone." The interaction creates an effect that is absent from both parents.

Apply this test before writing anything:
  1. Name the specific physical/chemical interaction between A's mechanism and B's mechanism.
  2. State the effect that interaction produces.
  3. Ask: is that effect present in A alone? In B alone? If it is present in either,
     you have an aggregation — not a combination.
  4. Ask: does one mechanism enable, amplify, or remove a limitation of the other?
     If neither does, they do not combine.

If A and B do NOT interact to yield a new effect, DO NOT invent a hybrid to satisfy this
request. Say so honestly by returning:
  {{"combination_viable": false, "reason": "1 sentence on why these two only aggregate"}}
Returning false is the correct answer when the mechanisms are merely compatible rather
than synergistic. A discarded non-combination is far more useful than a plausible-sounding
aggregation.

If they DO combine, output the combined invention in the standard schema, and make the
summary state the new technical effect explicitly — name the interaction, and say what
neither parent achieves alone."""


async def _evolve_one(
    strategy: str,
    prompt: str,
    session_id: str,
    parent: Invention | None = None,
) -> Invention | None:
    messages = [{"role": "user", "content": prompt}]
    try:
        text, _ = await run_claude_with_tools(
            model=settings.r_evolution_model,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=[],
            max_tool_rounds=1,
        )
    except AgentCallError as e:
        logger.error(f"Evolution {strategy} failed: {e}")
        return None

    try:
        data = parse_json_response(text)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Evolution parse failed for {strategy}: {e}")
        return None

    # The combine step may legitimately decline: if the two parent mechanisms only
    # aggregate (each keeps doing its own job) rather than interacting to yield a new
    # technical effect, there is no invention to make. Dropping the candidate is the
    # correct outcome — better than an obvious collocation seeded at parent Elo + 50.
    if data.get("combination_viable") is False:
        logger.info(
            f"Evolution {strategy} declined — no new technical effect: "
            f"{data.get('reason', 'no reason given')}"
        )
        return None

    return Invention(
        session_id=session_id,
        title=data.get("title", "Evolved Invention"),
        summary=data.get("summary", ""),
        mechanism=data.get("mechanism", ""),
        strategy=strategy,
        # Carry the parent's trigger provenance through to the evolved variant
        trigger_advance=parent.trigger_advance if parent else "",
        trigger_source_domain=parent.trigger_source_domain if parent else "",
        trigger_url=parent.trigger_url if parent else "",
        # No elo_score here on purpose. A variant enters the tournament at the same
        # rating as everything else and has to win matches to place above its parent.
        # The paper is explicit that this is what makes speculative evolution safe:
        # "each new hypothesis must also compete in the tournament".
    )


def _cluster_of(inv_id: str, clusters: list[dict]) -> str | None:
    """Which mechanistic family (proximity cluster) an invention belongs to."""
    for c in clusters:
        if inv_id in c.get("invention_ids", []):
            return c.get("label")
    return None


def _combination_candidates(
    top_k: list[Invention],
    clusters: list[dict],
) -> list[tuple[Invention, Invention]]:
    """
    Choose which pairs are worth attempting to combine.

    Elo rank says how good each invention is ALONE — it says nothing about whether two
    mechanisms interact. So rank alone is the wrong basis for pairing: the top two are
    often simply the two best independent ideas, and when they sit in the same
    mechanistic family they are near-variants, the worst possible combination candidates.

    Pairing therefore uses proximity's cluster assignments: only pairs drawn from
    DIFFERENT mechanistic families are attempted, because a new technical effect has to
    come from unlike mechanisms interacting. Rank is kept only as a tiebreak — given two
    equally cross-family pairs, prefer the stronger parents.

    Returns [] when nothing is worth attempting; combining is not mandatory.
    """
    pairs = [
        (top_k[i], top_k[j])
        for i in range(len(top_k))
        for j in range(i + 1, len(top_k))
    ]
    if not pairs:
        return []

    # Without cluster data we cannot tell variants from genuinely different approaches.
    # Fall back to the single best-ranked pair rather than guessing at more.
    if not clusters:
        logger.info("Evolution: no cluster data — attempting only the top-ranked pair")
        return pairs[:1]

    cross_family = []
    for a, b in pairs:
        ca, cb = _cluster_of(a.id, clusters), _cluster_of(b.id, clusters)
        if ca is not None and cb is not None and ca == cb:
            logger.info(
                f"Evolution: skipping pair in one family ({ca}) — "
                f"'{a.title[:28]}' + '{b.title[:28]}' are variants, not complements"
            )
            continue
        cross_family.append((a, b))

    if not cross_family:
        logger.info(
            "Evolution: no combination attempted — every top-ranked pair shares a "
            "mechanistic family, so no pair can yield a new technical effect"
        )
        return []

    # Strongest parents first, then cap: each attempt is a model call, and a third-best
    # pair is unlikely to beat what the first two produce.
    cross_family.sort(key=lambda p: p[0].elo_score + p[1].elo_score, reverse=True)
    return cross_family[:MAX_COMBINATION_ATTEMPTS]


async def run(
    ranked_inventions: list[Invention],
    reviews: dict[str, Review],
    problem_statement: str,
    session_id: str,
    on_progress: Callable,
    clusters: list[dict] | None = None,
) -> EvolutionResult:
    """
    Evolve top-K inventions.

    - Enhance each top-K invention individually
    - Attempt combination only for cross-family pairs that may yield a new technical
      effect; each attempt may still decline. Zero combinations is a valid outcome.
    """
    top_k = ranked_inventions[:settings.top_k_for_evolution]

    if not top_k:
        return EvolutionResult(evolved_inventions=[], final_ranked=ranked_inventions)

    tasks = []

    # Enhancement tasks — one per top-K invention
    for inv in top_k:
        review = reviews.get(inv.id)
        tasks.append(
            _evolve_one(
                strategy="enhanced",
                prompt=_enhance_prompt(inv, review, problem_statement),
                session_id=session_id,
                parent=inv,
            )
        )

    # Combination tasks — only pairs that could plausibly yield a new technical effect
    for inv_a, inv_b in _combination_candidates(top_k, clusters or []):
        # Attribute the hybrid to A's trigger, falling back to B's if A has none
        # (only literature_exploration inventions carry a trigger).
        trigger_parent = inv_a if inv_a.trigger_advance else inv_b
        tasks.append(
            _evolve_one(
                strategy="combined",
                prompt=_combine_prompt(inv_a, inv_b, problem_statement),
                session_id=session_id,
                parent=trigger_parent,
            )
        )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    evolved = []
    for r in results:
        if isinstance(r, Exception):
            logger.error(f"Evolution task error: {r}")
        elif r is not None:
            evolved.append(r)
            await on_progress("evolution_complete", {
                "invention_id": r.id,
                "title": r.title,
                "strategy": r.strategy,
                "elo_seed": r.elo_score,
            })

    # The merged field, deliberately NOT sorted here: these variants have not competed
    # yet. The caller reviews them and re-runs the tournament, which is what decides
    # the final order.
    all_inventions = ranked_inventions + evolved

    logger.info(
        f"Evolution complete: {len(evolved)} variant(s) produced, "
        f"pending review and tournament entry"
    )

    return EvolutionResult(
        evolved_inventions=evolved,
        final_ranked=all_inventions,
    )
