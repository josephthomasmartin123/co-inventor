"""
Trigger Discovery Agent

This is the most novel part of Co-Inventor — it actively hunts for "invention triggers":
recent advances or breakthroughs that enable new invention opportunities.

The core insight: the best inventions are often created by applying a recent advance
from one domain to solve a problem in a different domain. For example:
- Nanobubble research from food science → applied to concrete workability
- CRISPR gene editing → applied to crop disease resistance
- Phase-change materials from space tech → applied to building insulation

This agent:
1. Decomposes the problem into underlying physical/chemical/engineering principles
2. Searches for recent advances (last ~5 years) in the problem domain
3. Searches for recent advances in adjacent and distant domains that solve similar physics
4. Returns a ranked list of triggers for the Generation agent to exploit
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable

from app.agents.base import run_claude_with_tools, parse_json_response, AgentCallError
from app.config import settings
from app.models.invention import Trigger
from app.models.pipeline import TriggerDiscoveryResult
from app.tools.web_search import WEB_SEARCH_TOOL

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a technology scout and invention trigger analyst.

Your job is to find "invention triggers" — recent advances, discoveries, or newly available
technologies that could enable novel inventions for a given technical problem.

The best triggers are:
- RECENT (ideally 2020-2026): New capabilities not available 10 years ago
- CROSS-DOMAIN: From a different field than the problem domain (this increases novelty odds)
- ENABLING: They provide a new mechanism that directly addresses a root cause of the problem
- SPECIFIC: A concrete material, process, or technology — not a vague trend

Classic example of an excellent trigger:
Problem: Making concrete more workable without adding water
Trigger: "Nanobubble technology in food processing (2021) — nanobubbles act as rheology
modifiers, reducing interfacial friction in viscous suspensions"
Why it works: Concrete's workability is limited by internal friction between particles.
Nanobubbles from food industry can reduce this friction without changing chemistry.

You have access to web_search. Use it strategically:
- Search for the core physics/chemistry principles behind the problem
- Search for recent breakthroughs in adjacent fields (materials science, biotech,
  nanotech, food science, aerospace, manufacturing)
- Search explicitly for "2022 2023 2024 2025 breakthrough [relevant principle]"
- Try at least 3-4 different search angles before concluding

Output ONLY a JSON array of trigger objects. No prose outside the JSON.
"""

def _build_user_prompt(problem_statement: str, n_triggers: int) -> str:
    return f"""PROBLEM TO SOLVE:
{problem_statement}

Find {n_triggers} distinct invention triggers for this problem.

PROCESS:
1. First, identify the 2-3 core physical/chemical/engineering challenges underlying this problem
2. For each challenge, search for recent advances that address the same underlying principle
3. Look in at least 2 distant domains (e.g., if the problem is mechanical, look at biotech, food science, etc.)

OUTPUT SCHEMA — return a JSON array of exactly {n_triggers} triggers:
[
  {{
    "advance": "Description of the recent advance or new technology",
    "source_domain": "The field where this advance originates (e.g. 'food science', 'aerospace')",
    "mechanism": "How this advance works physically/chemically — what does it actually do?",
    "relevance": "Why this could help solve the stated problem — be specific about the link",
    "source_url": "URL of a paper/article describing this advance (from your web search)",
    "recency": "Approximate year or year range (e.g. '2022' or '2021-2023')",
    "distance": "same_domain | adjacent | distant"
  }}
]

Prioritise: distant-domain triggers first, then adjacent, then same-domain.
All {n_triggers} triggers must be distinct — different advances, not variations of one idea."""


async def run(
    problem_statement: str,
    session_id: str,
    on_progress: Callable,
    n_triggers: int | None = None,
) -> TriggerDiscoveryResult:
    """
    Discover invention triggers for the given problem.

    Runs a single Claude call (with web search tools) to find recent advances
    across multiple domains that could enable novel inventions.
    """
    if n_triggers is None:
        n_triggers = settings.trigger_count

    user_prompt = _build_user_prompt(problem_statement, n_triggers)

    try:
        text, search_results = await run_claude_with_tools(
            model=settings.r_trigger_model,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            tools=[WEB_SEARCH_TOOL],
            max_tool_rounds=8,  # Allow more rounds — trigger discovery is search-heavy
        )
    except AgentCallError as e:
        logger.error(f"Trigger discovery failed: {e}")
        return TriggerDiscoveryResult(triggers=[])

    # Parse response
    triggers = []
    try:
        raw = parse_json_response(text)
        if isinstance(raw, list):
            trigger_data = raw
        elif isinstance(raw, dict) and "triggers" in raw:
            trigger_data = raw["triggers"]
        else:
            trigger_data = []

        for item in trigger_data[:n_triggers]:
            if not isinstance(item, dict):
                continue
            trigger = Trigger(
                session_id=session_id,
                advance=item.get("advance", ""),
                source_domain=item.get("source_domain", ""),
                mechanism=item.get("mechanism", ""),
                relevance=item.get("relevance", ""),
                source_url=item.get("source_url", ""),
                recency=item.get("recency", ""),
                distance=item.get("distance", "adjacent"),
            )
            triggers.append(trigger)
            await on_progress("trigger_found", {
                "trigger_id": trigger.id,
                "advance": trigger.advance[:80],
                "source_domain": trigger.source_domain,
                "distance": trigger.distance,
            })

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse trigger discovery response: {e}\nRaw: {text[:500]}")

    logger.info(f"Trigger discovery: found {len(triggers)} triggers")
    return TriggerDiscoveryResult(triggers=triggers)
