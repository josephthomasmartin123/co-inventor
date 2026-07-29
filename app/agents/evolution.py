"""
Evolution Agent

Takes the top-ranked inventions and produces improved variants:

1. ENHANCE — Deepen the mechanism: add specificity, address reflection weaknesses,
   strengthen the non-obviousness argument. Produces one enhanced variant per top-K invention.

2. COMBINE — Merge the top-2 inventions into a hybrid that captures the best of both.
   Often produces the strongest final candidate.

Evolved inventions seed their Elo at parent score + 50 (optimistic prior).
They are merged with the original ranked list and sorted to produce final top-5.
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
    parent_elo: float,
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
        elo_score=parent_elo + 50.0,    # Optimistic seed — evolved versions are expected to be better
    )


async def run(
    ranked_inventions: list[Invention],
    reviews: dict[str, Review],
    problem_statement: str,
    session_id: str,
    on_progress: Callable,
) -> EvolutionResult:
    """
    Evolve top-K inventions.

    - Enhance each top-K individually
    - Combine top-1 and top-2 if both exist
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
                parent_elo=inv.elo_score,
                parent=inv,
            )
        )

    # Combination task — merge #1 + #2
    if len(top_k) >= 2:
        avg_elo = (top_k[0].elo_score + top_k[1].elo_score) / 2
        # Attribute the hybrid to #1's trigger, falling back to #2's if #1 has none
        # (only literature_exploration inventions carry a trigger).
        trigger_parent = top_k[0] if top_k[0].trigger_advance else top_k[1]
        tasks.append(
            _evolve_one(
                strategy="combined",
                prompt=_combine_prompt(top_k[0], top_k[1], problem_statement),
                session_id=session_id,
                parent_elo=avg_elo,
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

    # Merge original + evolved, sort by Elo
    all_inventions = ranked_inventions + evolved
    all_inventions.sort(key=lambda x: x.elo_score, reverse=True)

    logger.info(
        f"Evolution complete: {len(evolved)} evolved. "
        f"Final top-3: {' | '.join(i.title[:25] for i in all_inventions[:3])}"
    )

    return EvolutionResult(
        evolved_inventions=evolved,
        final_ranked=all_inventions,
    )
