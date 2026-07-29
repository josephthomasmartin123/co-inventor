"""
Meta-review Agent — aligned with Co-Scientist paper (Section 3.5)

The paper: "The meta-review agent synthesizes the insights from all scientist reviews
in order to identify recurring themes, surface previously unnoticed details, and
generate feedback that can guide subsequent generation and reflection agents."

Concretely, the meta-review agent:
  - Reads ALL reviews (not just the top-ranked inventions)
  - Identifies recurring patterns: what mechanisms consistently scored well/poorly
  - Spots missed opportunities: what the reflection agent may have overlooked
  - Produces a "research overview" — a structured synthesis of what was learned
  - In iterative systems, its output feeds back into the next generation round

In our single-pass pipeline, the meta-review output is shown to the user as a
"Research Overview" section — the system's synthesis of what it learned about
the problem space through the full pipeline run.
"""
from __future__ import annotations

import json
import logging
from typing import Callable

from app.agents.base import run_claude_with_tools, parse_json_response, AgentCallError
from app.config import settings
from app.models.invention import Invention, Review
from app.models.pipeline import MetaReviewResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior research director synthesising insights from a
multi-agent invention research session.

You have access to all the invention concepts generated and all their evaluations.
Your job is to step back and see the whole picture:
  - What approaches worked well? What mechanisms scored consistently high?
  - What fundamental challenges keep appearing across evaluations?
  - What promising directions were explored but not fully developed?
  - What cross-domain connections or insights emerged from the process?
  - What would you recommend as the single most promising direction to pursue further?

Write for a technically sophisticated reader (patent strategist or R&D lead).
Be specific — name mechanisms, not vague categories.

Output ONLY valid JSON. No prose outside the JSON."""


def _build_prompt(
    problem: str,
    ranked_inventions: list[Invention],
    reviews: dict[str, Review],
) -> str:
    # Build a compact representation of all inventions + their scores
    inv_summaries = []
    for i, inv in enumerate(ranked_inventions):
        r = reviews.get(inv.id)
        score_str = (
            f"N:{r.novelty_score} P:{r.patentability_score} F:{r.feasibility_score} "
            f"Fit:{r.problem_fit_score} Overall:{r.overall_score:.2f}"
            if r else "No review"
        )
        inv_summaries.append(
            f"#{i+1} [{inv.strategy}] {inv.title}\n"
            f"   Mechanism: {inv.mechanism[:180]}\n"
            f"   Scores: {score_str}"
            + (f"\n   Review note: {r.novelty_rationale[:120]}" if r else "")
        )

    inv_block = "\n\n".join(inv_summaries)

    return f"""PROBLEM BEING SOLVED:
{problem}

ALL INVENTIONS (ranked by Elo, best first):
{inv_block}

Synthesise insights across ALL inventions and ALL evaluations.

OUTPUT SCHEMA:
{{
  "overview": "3-4 sentences: what this problem space looks like after exploring it deeply. What is the nature of the challenge? What solution families exist?",
  "strongest_approaches": [
    "Specific mechanism or approach that consistently scored well — 1-2 sentences each"
  ],
  "recurring_challenges": [
    "Technical barrier that kept appearing across evaluations — 1-2 sentences each"
  ],
  "unexplored_directions": [
    "Promising avenue not well covered by the inventions generated — 1-2 sentences each"
  ],
  "cross_domain_insight": "The most interesting cross-domain connection or analogy that emerged — 1-2 sentences",
  "recommendation": "Single most promising direction to pursue: specific mechanism + why — 2-3 sentences"
}}"""


async def run(
    problem_statement: str,
    ranked_inventions: list[Invention],
    reviews: dict[str, Review],
    session_id: str,
    on_progress: Callable,
) -> MetaReviewResult:
    """
    Synthesise all evaluations into a research overview.
    Run after ranking, using the full ranked list (not just top-5).
    """
    if not ranked_inventions:
        return MetaReviewResult(
            overview="No inventions to synthesise.",
            strongest_approaches=[],
            recurring_challenges=[],
            unexplored_directions=[],
            cross_domain_insight="",
            recommendation="",
        )

    messages = [{"role": "user", "content": _build_prompt(problem_statement, ranked_inventions, reviews)}]

    try:
        text, _ = await run_claude_with_tools(
            model=settings.r_evolution_model,   # strong model for synthesis
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=[],
            max_tool_rounds=1,
            agent_name="meta_review",
        )
    except AgentCallError as e:
        logger.error(f"Meta-review failed: {e}")
        return MetaReviewResult(
            overview="Meta-review could not complete.",
            strongest_approaches=[],
            recurring_challenges=[],
            unexplored_directions=[],
            cross_domain_insight="",
            recommendation="",
        )

    try:
        data = parse_json_response(text)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Meta-review parse failed: {e}")
        # Return raw text as overview if parse fails
        return MetaReviewResult(
            overview=text[:500],
            strongest_approaches=[],
            recurring_challenges=[],
            unexplored_directions=[],
            cross_domain_insight="",
            recommendation="",
        )

    result = MetaReviewResult(
        overview=data.get("overview", ""),
        strongest_approaches=data.get("strongest_approaches", []),
        recurring_challenges=data.get("recurring_challenges", []),
        unexplored_directions=data.get("unexplored_directions", []),
        cross_domain_insight=data.get("cross_domain_insight", ""),
        recommendation=data.get("recommendation", ""),
    )

    await on_progress("meta_review_complete", {
        "overview_preview": result.overview[:120],
        "recommendation_preview": result.recommendation[:100],
    })

    logger.info("Meta-review complete")
    return result
