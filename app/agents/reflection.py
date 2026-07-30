"""
Reflection Agent — aligned with Co-Scientist paper (Section 3.2)

Five evaluation dimensions. The paper's criteria are "plausibility, novelty, testability,
and safety" plus alignment with the research goal; testability becomes feasibility,
patentability is added and safety is not scored — see docs/ADAPTATION.md §1.2 and §3.4:
  1. Novelty             (0.30) — non-obvious mechanism, absent from prior art
  2. Scientific plausibility (0.25) — underlying physics/chemistry is sound
  3. Patentability       (0.20) — specific, claimable technical mechanism
  4. Feasibility         (0.15) — buildable with current/near-future technology
  5. Problem fit         (0.10) — addresses root cause, not a symptom

Two-tier review:
  Tier 1 — Initial Review: quick pass, no web search. NB: a concept that fails is scored
           down, not removed — it still enters the tournament (docs/ADAPTATION.md §2.4).
  Tier 2 — Full Review:    comprehensive evaluation with prior art search.

Also called on evolved variants in stage 5b, so this must stay safe to run on a subset.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable

from app.agents.base import run_claude_with_tools, parse_json_response, AgentCallError
from app.config import settings
from app.models.invention import Invention, Review
from app.models.pipeline import ReflectionResult
from app.tools.web_search import WEB_SEARCH_TOOL

logger = logging.getLogger(__name__)
_SEMAPHORE = asyncio.Semaphore(4)

# ── Scoring weights — must sum to 1.0 ─────────────────────────────────────
WEIGHTS = {
    "novelty":                 0.30,
    "scientific_plausibility": 0.25,
    "patentability":           0.20,
    "feasibility":             0.15,
    "problem_fit":             0.10,
}

# ── Tier 1: Initial Review ────────────────────────────────────────────────

_INITIAL_SYSTEM = """You are a senior patent examiner doing a rapid first-pass review.
Score each dimension 1-5 without doing a literature search. Be calibrated: 3 = genuinely average.

If the invention brings together known elements, check that they INTERACT to produce a
technical effect neither produces alone. Features sitting side by side, each doing its own
separate job, are an aggregation — score novelty and patentability low and fail the filter.

Output ONLY valid JSON."""

def _initial_prompt(problem: str, inv: Invention) -> str:
    return f"""PROBLEM: {problem}

INVENTION:
Title: {inv.title}
Summary: {inv.summary}
Mechanism: {inv.mechanism}

Quick assessment — no search needed. Score all five dimensions.

{{
  "novelty_score": <int 1-5>,
  "novelty_rationale": "1 sentence",
  "scientific_plausibility_score": <int 1-5>,
  "scientific_plausibility_rationale": "1 sentence — does the physics/chemistry work?",
  "feasibility_score": <int 1-5>,
  "feasibility_rationale": "1 sentence",
  "problem_fit_score": <int 1-5>,
  "problem_fit_rationale": "1 sentence",
  "patentability_score": <int 1-5>,
  "patentability_rationale": "1 sentence",
  "initial_pass": true or false,
  "filter_reason": "If false: 1 sentence on why too weak. If true: empty string."
}}"""

# ── Tier 2: Full Review ───────────────────────────────────────────────────

_FULL_SYSTEM = """You are a patent examiner and technical feasibility expert.

Evaluate thoroughly on five dimensions:

NOVELTY (0.30 weight): Use web_search to check prior art. Search the specific mechanism.
  A high score requires the mechanism to be absent from prior art.
  AGGREGATION TEST — apply this whenever the invention brings together known elements:
    absence from prior art is NOT novelty if the pairing is an obvious collocation.
    Ask whether the elements INTERACT to produce a technical effect that neither
    produces alone. If each element merely continues doing its own job side by side,
    that is an aggregation: score 1-2 on novelty even if no single document shows the
    exact pairing, and say so in the rationale. Reserve 4-5 for combinations where you
    can name the interaction and the effect that emerges only from it.

SCIENTIFIC PLAUSIBILITY (0.25 weight): Does the underlying science work?
  Are the physical, chemical, or biological principles sound?
  A high score means a domain expert would say "yes, this should work in principle."

PATENTABILITY (0.20 weight): Is it a specific, claimable technical solution?
  Not an abstract idea or desired result — a concrete mechanism with identifiable elements.
  A claim reciting features that do not functionally interact is an aggregation and
  cannot support an inventive step — score it 1-2 regardless of how specific the
  individual features are.

FEASIBILITY (0.15 weight): Can it be built with current or near-future (5-year) technology?

PROBLEM FIT (0.10 weight): Does the mechanism address the root cause of the problem,
  not just a symptom?

Use at least 2 web searches: one for the mechanism, one for adjacent prior art.
Output ONLY valid JSON."""

def _full_prompt(problem: str, inv: Invention) -> str:
    return f"""PROBLEM: {problem}

INVENTION:
Title: {inv.title}
Summary: {inv.summary}
Mechanism: {inv.mechanism}
Strategy: {inv.strategy}

Search for prior art, then score all five dimensions.

{{
  "novelty_score": <int 1-5>,
  "novelty_rationale": "1-2 sentences. Cite conflicting prior art URLs if found.",
  "prior_art_found": ["<url>", ...],
  "scientific_plausibility_score": <int 1-5>,
  "scientific_plausibility_rationale": "1-2 sentences — is the science sound?",
  "feasibility_score": <int 1-5>,
  "feasibility_rationale": "1-2 sentences.",
  "problem_fit_score": <int 1-5>,
  "problem_fit_rationale": "1-2 sentences — how does the mechanism address the root cause?",
  "patentability_score": <int 1-5>,
  "patentability_rationale": "1-2 sentences — what is the specific claimable element?"
}}"""

# ── Build Review from parsed data ─────────────────────────────────────────

def _build_review(data: dict, inv: Invention, session_id: str,
                  extra_urls: list[str] | None = None) -> Review:
    def clamp(v, default=3): return max(1, min(5, int(v))) if v is not None else default

    novelty      = clamp(data.get("novelty_score"))
    sci_plaus    = clamp(data.get("scientific_plausibility_score"))
    feasibility  = clamp(data.get("feasibility_score"))
    fit          = clamp(data.get("problem_fit_score"))
    patentability= clamp(data.get("patentability_score"))

    prior_art = list(data.get("prior_art_found", []))
    for url in (extra_urls or []):
        if url and url not in prior_art:
            prior_art.append(url)

    overall = round(
        novelty      * WEIGHTS["novelty"]
        + sci_plaus  * WEIGHTS["scientific_plausibility"]
        + patentability * WEIGHTS["patentability"]
        + feasibility * WEIGHTS["feasibility"]
        + fit        * WEIGHTS["problem_fit"],
        3,
    )

    return Review(
        invention_id=inv.id,
        session_id=session_id,
        novelty_score=novelty,
        novelty_rationale=data.get("novelty_rationale", ""),
        prior_art_found=prior_art[:6],
        scientific_plausibility_score=sci_plaus,
        scientific_plausibility_rationale=data.get("scientific_plausibility_rationale", ""),
        feasibility_score=feasibility,
        feasibility_rationale=data.get("feasibility_rationale", ""),
        problem_fit_score=fit,
        problem_fit_rationale=data.get("problem_fit_rationale", ""),
        patentability_score=patentability,
        patentability_rationale=data.get("patentability_rationale", ""),
        overall_score=overall,
    )

# ── Per-invention evaluation ──────────────────────────────────────────────

async def _evaluate_one(inv: Invention, problem: str,
                        session_id: str, on_progress: Callable) -> Review | None:
    async with _SEMAPHORE:

        # Tier 1: quick filter
        init_msgs = [{"role": "user", "content": _initial_prompt(problem, inv)}]
        try:
            init_text, _ = await run_claude_with_tools(
                model=settings.r_reflection_model, system=_INITIAL_SYSTEM,
                messages=init_msgs, tools=[], max_tool_rounds=1,
                agent_name=f"reflection/initial/{inv.id[:6]}",
            )
            init_data = parse_json_response(init_text)
        except Exception as e:
            logger.warning(f"Initial review error {inv.id[:8]}: {e}")
            init_data = {"initial_pass": True}

        if not init_data.get("initial_pass", True):
            reason = init_data.get("filter_reason", "Filtered by initial review")
            logger.info(f"Filtered: {inv.title[:40]} — {reason}")
            review = _build_review({**init_data, "prior_art_found": []}, inv, session_id)
            await on_progress("review_complete", {
                "invention_id": inv.id, "invention_title": inv.title,
                "tier": "initial", "passed": False, "filter_reason": reason,
                "overall_score": review.overall_score,
            })
            return review

        # Tier 2: full review with web search
        async def _on_search(agent: str, query: str, results: list) -> None:
            await on_progress("search_query", {
                "agent": agent, "query": query,
                "result_count": len(results),
                "top_results": [{"title": r.get("title",""), "url": r.get("url","")}
                                for r in results[:3]],
            })

        full_msgs = [{"role": "user", "content": _full_prompt(problem, inv)}]
        try:
            full_text, search_results = await run_claude_with_tools(
                model=settings.r_reflection_model, system=_FULL_SYSTEM,
                messages=full_msgs, tools=[WEB_SEARCH_TOOL],
                on_tool_call=_on_search,
                agent_name=f"reflection/full/{inv.id[:6]}",
                max_tool_rounds=5,
            )
            full_data = parse_json_response(full_text)
        except Exception as e:
            logger.warning(f"Full review error {inv.id[:8]}: {e}. Using initial scores.")
            full_data = init_data
            search_results = []

        extra_urls = [r.get("url","") for r in search_results if r.get("url")]
        review = _build_review(full_data, inv, session_id, extra_urls)

        await on_progress("review_complete", {
            "invention_id": inv.id, "invention_title": inv.title,
            "tier": "full", "passed": True,
            "novelty_score": review.novelty_score,
            "scientific_plausibility_score": review.scientific_plausibility_score,
            "patentability_score": review.patentability_score,
            "overall_score": review.overall_score,
            "prior_art_count": len(review.prior_art_found),
        })
        return review

# ── Main entry point ──────────────────────────────────────────────────────

async def run(inventions: list[Invention], problem_statement: str,
              session_id: str, on_progress: Callable) -> ReflectionResult:
    tasks = [_evaluate_one(inv, problem_statement, session_id, on_progress)
             for inv in inventions]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    reviews = [r for r in results if not isinstance(r, Exception) and r is not None]
    logger.info(f"Reflection: {len(reviews)}/{len(inventions)} reviewed")
    return ReflectionResult(reviews=reviews)
