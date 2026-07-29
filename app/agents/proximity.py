"""
Proximity Agent — aligned with Co-Scientist paper (Section 3.4)

The paper: "The proximity agent computes the semantic similarity between research
hypotheses. This enables clustering of related ideas, de-duplication, and ensures
diverse exploration of the problem space."

This agent runs AFTER generation and BEFORE reflection, for two purposes:

1. DE-DUPLICATION: If two inventions describe essentially the same mechanism
   (different words, same physics), drop the weaker one. Reflection is expensive
   (web search per invention) — no point evaluating duplicates.

2. DIVERSITY CHECK: Flag if all inventions cluster around one approach, which
   would suggest the generation agent got stuck. (Logged but not acted on in v1.)

Implementation: single Claude call with all invention titles + summaries.
No web search needed — this is pure semantic comparison.
"""
from __future__ import annotations

import json
import logging
from typing import Callable

from app.agents.base import run_claude_with_tools, parse_json_response, AgentCallError
from app.config import settings
from app.models.invention import Invention
from app.models.pipeline import ProximityResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a patent examiner expert in identifying duplicate or overlapping inventions.

Your task: given a list of invention concepts, identify near-duplicates and cluster related ideas.

Two inventions are near-duplicates if they describe essentially the same core mechanism —
even if described with different words or from different angles.
Example of near-duplicates: "phase-change microchannel cooler" and "evaporative microfluidic
heat spreader" are essentially the same mechanism.
Example of NOT duplicates: "nanobubble drag reduction" and "hydrophobic surface coating"
are different mechanisms that happen to address the same problem.

Be conservative: only flag as duplicate if the mechanism is genuinely the same.
When in doubt, keep both.

Output ONLY valid JSON. No prose outside the JSON."""


def _build_prompt(inventions: list[Invention]) -> str:
    inv_list = "\n".join(
        f'{i+1}. ID={inv.id[:8]}\n   Title: {inv.title}\n   Mechanism: {inv.mechanism[:200]}'
        for i, inv in enumerate(inventions)
    )
    return f"""INVENTIONS TO ANALYSE:

{inv_list}

TASK:
1. Identify any near-duplicate pairs (same core mechanism, different words).
   For each duplicate pair, keep the one with the more specific/detailed mechanism.
2. Assign each invention to a cluster (A, B, C, ...) based on the underlying approach.
   Inventions in the same cluster share a mechanistic family.

OUTPUT SCHEMA:
{{
  "keep": ["<full-id-1>", "<full-id-2>", ...],   // IDs to keep (de-duplicated)
  "remove": [                                      // near-duplicates to drop
    {{"id": "<full-id>", "duplicate_of": "<full-id>", "reason": "1 sentence"}}
  ],
  "clusters": [
    {{"label": "A", "theme": "brief description", "invention_ids": ["<id>", ...]}}
  ]
}}

Include ALL invention IDs in either keep or remove. Never omit one."""


async def run(
    inventions: list[Invention],
    session_id: str,
    on_progress: Callable,
) -> ProximityResult:
    """
    Deduplicate and cluster inventions.
    Returns filtered list (duplicates removed) and cluster assignments.
    """
    if len(inventions) <= 2:
        # Nothing to deduplicate
        return ProximityResult(
            inventions=inventions,
            removed_ids=[],
            clusters=[],
        )

    messages = [{"role": "user", "content": _build_prompt(inventions)}]

    try:
        text, _ = await run_claude_with_tools(
            model=settings.r_generation_model,   # fast model is fine for this
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=[],
            max_tool_rounds=1,
            agent_name="proximity",
        )
    except AgentCallError as e:
        logger.error(f"Proximity agent failed: {e}")
        return ProximityResult(inventions=inventions, removed_ids=[], clusters=[])

    # Parse response
    try:
        data = parse_json_response(text)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Proximity parse failed: {e}. Keeping all inventions.")
        return ProximityResult(inventions=inventions, removed_ids=[], clusters=[])

    # Build id → invention map for lookup
    inv_map = {inv.id: inv for inv in inventions}

    # Determine which to remove
    remove_ids: set[str] = set()
    remove_details = data.get("remove", [])
    for item in remove_details:
        if isinstance(item, dict):
            rid = item.get("id", "")
            # Match partial IDs (we showed 8-char prefixes)
            for full_id in inv_map:
                if full_id.startswith(rid) or rid.startswith(full_id[:8]):
                    remove_ids.add(full_id)
                    logger.info(
                        f"Proximity: removing duplicate '{inv_map.get(full_id, {}).title if full_id in inv_map else rid}' "
                        f"— {item.get('reason', '')}"
                    )
                    break

    filtered = [inv for inv in inventions if inv.id not in remove_ids]
    clusters = data.get("clusters", [])

    await on_progress("proximity_complete", {
        "total_in": len(inventions),
        "total_out": len(filtered),
        "removed_count": len(remove_ids),
        "cluster_count": len(clusters),
        "clusters": [
            {"label": c.get("label", ""), "theme": c.get("theme", "")}
            for c in clusters
        ],
    })

    logger.info(
        f"Proximity: {len(inventions)} → {len(filtered)} "
        f"({len(remove_ids)} removed, {len(clusters)} clusters)"
    )
    return ProximityResult(inventions=filtered, removed_ids=list(remove_ids), clusters=clusters)
