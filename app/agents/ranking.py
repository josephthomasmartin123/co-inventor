"""
Ranking Agent — Elo Tournament

Ranks inventions using an Elo-based pairwise tournament, borrowed from chess ratings.

Why Elo tournament vs. simple score sorting?
- Absolute scores from the Reflection agent have calibration biases
- Pairwise comparison by a "fresh" judge (no prior context) is more reliable
- Debate format forces Claude to articulate WHY one is better — improves signal quality
- Elo naturally handles transitivity: if A > B > C, A's score stays high

Tournament design:
- Seed Elo from reflection overall_score (better start than flat 1000)
- K_FACTOR=32: moderate sensitivity, appropriate for small population
- 12 rounds with pair-repeat avoidance
- No web search — pure comparative reasoning (fast ~1-2s per debate)
"""
from __future__ import annotations

import logging
import math
import random
from typing import Callable

from app.agents.base import run_claude_with_tools, parse_json_response, AgentCallError
from app.config import settings
from app.models.invention import EloMatchup, Invention, Review
from app.models.pipeline import RankingResult

logger = logging.getLogger(__name__)

K_FACTOR = 32
INITIAL_ELO = 1000.0

SYSTEM_PROMPT = """You are a senior patent attorney, technology investor, and domain-agnostic
technical expert. Compare two invention concepts and decide which is more valuable.

Evaluation criteria (in priority order):
1. NOVELTY — Which has a more non-obvious, surprise-worthy mechanism?
2. MECHANISM SPECIFICITY — Which is more concretely defined (not vague outcomes)?
   Penalise aggregation: an invention that lists several features which do not
   functionally interact is weaker than a single mechanism producing a clear technical
   effect, even though it sounds richer. More features is not more invention — ask what
   effect emerges from the parts working together, and prefer the one that has one.
3. COMMERCIAL POTENTIAL — Which would be harder for competitors to design around?
4. FEASIBILITY — Which is more achievable with near-term technology?
5. PROBLEM FIT — Which more directly addresses the core problem?

Be decisive — one invention must win. Do not declare a tie.

Output ONLY valid JSON:
{
  "winner": "A" or "B",
  "rationale": "2-3 sentences explaining the decision. Be specific — name the mechanism that won."
}"""


def _build_debate_prompt(
    problem: str,
    inv_a: Invention,
    inv_b: Invention,
    review_a: Review | None,
    review_b: Review | None,
) -> str:
    def score_line(r: Review | None) -> str:
        if r is None:
            return "No review available"
        return (
            f"Novelty {r.novelty_score}/5 | Feasibility {r.feasibility_score}/5 | "
            f"Fit {r.problem_fit_score}/5 | Patentability {r.patentability_score}/5"
        )

    return f"""PROBLEM: {problem}

INVENTION A:
Title: {inv_a.title}
Summary: {inv_a.summary}
Mechanism: {inv_a.mechanism}
Strategy: {inv_a.strategy}
Scores: {score_line(review_a)}

INVENTION B:
Title: {inv_b.title}
Summary: {inv_b.summary}
Mechanism: {inv_b.mechanism}
Strategy: {inv_b.strategy}
Scores: {score_line(review_b)}

Which invention is more valuable? Consider novelty, specificity, and commercial potential."""


def elo_update(winner_score: float, loser_score: float) -> tuple[float, float]:
    """Standard Elo update formula."""
    expected_winner = 1.0 / (1.0 + math.pow(10, (loser_score - winner_score) / 400.0))
    expected_loser = 1.0 - expected_winner
    new_winner = winner_score + K_FACTOR * (1.0 - expected_winner)
    new_loser = loser_score + K_FACTOR * (0.0 - expected_loser)
    return new_winner, new_loser


def _pick_pair(
    inventions: list[Invention],
    played_pairs: set[frozenset],
) -> tuple[Invention, Invention] | None:
    """Pick a pair not yet played. Returns None if all pairs exhausted."""
    candidates = []
    for i in range(len(inventions)):
        for j in range(i + 1, len(inventions)):
            pair = frozenset([inventions[i].id, inventions[j].id])
            if pair not in played_pairs:
                candidates.append((inventions[i], inventions[j]))
    if not candidates:
        return None
    return random.choice(candidates)


async def _debate(
    inv_a: Invention,
    inv_b: Invention,
    review_a: Review | None,
    review_b: Review | None,
    problem: str,
) -> tuple[str, str]:
    """
    Run a single pairwise debate. Returns (winner_id, rationale).
    No web search — pure comparative reasoning.
    """
    prompt = _build_debate_prompt(problem, inv_a, inv_b, review_a, review_b)
    messages = [{"role": "user", "content": prompt}]

    try:
        text, _ = await run_claude_with_tools(
            model=settings.r_ranking_model,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=[],       # No web search — keep debates fast
            max_tool_rounds=1,
            agent_name="ranking/debate",
        )
    except AgentCallError:
        # On failure, randomly assign winner to keep tournament moving
        return random.choice([inv_a.id, inv_b.id]), "Unable to evaluate (API error)"

    try:
        data = parse_json_response(text)
        winner_letter = str(data.get("winner", "A")).strip().upper()
        rationale = data.get("rationale", "")
        winner_id = inv_a.id if winner_letter == "A" else inv_b.id
        return winner_id, rationale
    except Exception:
        # Default to A on parse failure
        return inv_a.id, "Parse failed — defaulting to A"


async def run(
    inventions: list[Invention],
    reviews: dict[str, Review],
    problem_statement: str,
    session_id: str,
    on_progress: Callable,
) -> RankingResult:
    """
    Run Elo tournament.

    Seed Elo from reflection scores for better starting signal.
    Run N rounds of pairwise debates, updating Elo after each.
    """
    if len(inventions) < 2:
        return RankingResult(ranked_inventions=inventions)

    # Seed Elo from reflection overall_score (range 800-1200 maps to score 0-5)
    for inv in inventions:
        review = reviews.get(inv.id)
        if review:
            inv.elo_score = 800.0 + (review.overall_score / 5.0) * 400.0
        else:
            inv.elo_score = INITIAL_ELO

    matchup_log: list[EloMatchup] = []
    played_pairs: set[frozenset] = set()
    n_rounds = settings.elo_rounds

    for round_num in range(n_rounds):
        pair = _pick_pair(inventions, played_pairs)
        if pair is None:
            # All pairs played — reset to allow repeats
            played_pairs.clear()
            pair = _pick_pair(inventions, played_pairs)
            if pair is None:
                break

        inv_a, inv_b = pair
        played_pairs.add(frozenset([inv_a.id, inv_b.id]))

        winner_id, rationale = await _debate(
            inv_a, inv_b,
            reviews.get(inv_a.id),
            reviews.get(inv_b.id),
            problem_statement,
        )

        # Update Elo scores
        if winner_id == inv_a.id:
            inv_a.elo_score, inv_b.elo_score = elo_update(inv_a.elo_score, inv_b.elo_score)
            inv_a.elo_wins += 1
            inv_b.elo_losses += 1
        else:
            inv_b.elo_score, inv_a.elo_score = elo_update(inv_b.elo_score, inv_a.elo_score)
            inv_b.elo_wins += 1
            inv_a.elo_losses += 1

        matchup_log.append(EloMatchup(
            invention_a_id=inv_a.id,
            invention_b_id=inv_b.id,
            winner_id=winner_id,
            rationale=rationale,
            round=round_num,
        ))

        await on_progress("matchup_complete", {
            "round": round_num + 1,
            "total_rounds": n_rounds,
            "winner_id": winner_id,
            "winner_title": inv_a.title if winner_id == inv_a.id else inv_b.title,
        })

    # Sort by Elo score descending
    inventions.sort(key=lambda x: x.elo_score, reverse=True)

    logger.info(
        f"Ranking complete: top 3 — "
        + " | ".join(f"{inv.title[:30]} ({inv.elo_score:.0f})" for inv in inventions[:3])
    )

    return RankingResult(ranked_inventions=inventions)
