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

Focus on:
- Making the MECHANISM more specific (add materials, dimensions, process steps, control algorithms)
- Addressing weaknesses identified in the evaluation
- Strengthening non-obviousness — add a secondary effect or unexpected benefit
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

TASK: Combine the best aspects of A and B into a single superior invention.

A good combination:
- Preserves the core strengths of each
- Creates an emergent benefit from their interaction
- Is NOT just "A + B" — the combination should create something new
- Must be technically coherent (the mechanisms must be compatible)

Output the combined invention in the standard schema."""


async def _evolve_one(
    strategy: str,
    prompt: str,
    session_id: str,
    parent_elo: float,
    parent_trigger_id: str | None = None,
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

    return Invention(
        session_id=session_id,
        title=data.get("title", "Evolved Invention"),
        summary=data.get("summary", ""),
        mechanism=data.get("mechanism", ""),
        strategy=strategy,
        trigger_id=parent_trigger_id,
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
                parent_trigger_id=inv.trigger_id,
            )
        )

    # Combination task — merge #1 + #2
    if len(top_k) >= 2:
        avg_elo = (top_k[0].elo_score + top_k[1].elo_score) / 2
        tasks.append(
            _evolve_one(
                strategy="combined",
                prompt=_combine_prompt(top_k[0], top_k[1], problem_statement),
                session_id=session_id,
                parent_elo=avg_elo,
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
