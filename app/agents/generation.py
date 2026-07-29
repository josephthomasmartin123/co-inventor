"""
Generation Agent — aligned with Co-Scientist paper (Section 3.1)

The paper describes four generation strategies:
  1. literature_exploration — web search for prior work and recent advances;
     generate hypotheses distinct from or directly exploiting what was found.
     This naturally surfaces "invention triggers" (recent cross-domain advances)
     without needing a separate discovery stage.
  2. simulated_debate — self-play between expert personas with different perspectives;
     debate converges to a synthesised strongest concept.
  3. iterative_assumptions — enumerate implicit assumptions in the conventional approach,
     invert or challenge each, build inventions around the most fertile inversions.
  4. research_expansion — first principles decomposition and cross-domain analogy.
     Implemented as two sub-strategies (direct + analogical) run concurrently.

All strategies run in parallel via asyncio.gather.
Only literature_exploration uses web search.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable, Optional

from app.agents.base import run_claude_with_tools, parse_json_response, AgentCallError
from app.config import settings
from app.models.invention import Invention
from app.models.pipeline import GenerationResult
from app.tools.web_search import WEB_SEARCH_TOOL

logger = logging.getLogger(__name__)

# ── Shared system prompt ──────────────────────────────────────────────────

_BASE_SYSTEM = """You are a senior patent inventor and technical problem-solver.

Your task: generate novel invention concepts for a given technical problem.

CRITICAL RULES:
- Every invention must describe a MECHANISM — not just a desired outcome.
  WEAK: "Improved thermal management for LEDs"
  STRONG: "Micro-channel heat spreader with phase-change fluid embedded in the LED
           substrate; capillary wicking passively distributes and evaporates coolant"
- Mechanisms must be technically feasible in principle.
- The invention should surprise a domain expert — not restate the obvious.
- Output ONLY valid JSON. No prose, no explanation outside the JSON.
"""

_OUTPUT_SCHEMA = """
Return a JSON array of exactly 2 inventions:
[
  {
    "title": "5-10 word noun phrase naming the inventive concept",
    "summary": "2-3 sentences: what it is and why it is non-obvious",
    "mechanism": "2-3 sentences: HOW it works — specific materials, physics, process steps"
  },
  { ... }
]"""

# Extended schema for literature_exploration — includes trigger provenance fields
_OUTPUT_SCHEMA_LITERATURE = """
Return a JSON array of exactly 2 inventions. Because this strategy is trigger-driven,
each invention MUST include the specific advance that inspired it:
[
  {
    "title": "5-10 word noun phrase naming the inventive concept",
    "summary": "2-3 sentences: what it is and why it is non-obvious",
    "mechanism": "2-3 sentences: HOW it works — specific materials, physics, process steps",
    "trigger_advance": "1 sentence: the specific recent advance or cross-domain breakthrough that makes this newly possible. Be concrete — name the material, process, or discovery.",
    "trigger_source_domain": "The field this advance comes from (e.g. 'food science', 'aerospace', 'biotech')",
    "trigger_url": "URL of the paper/article/patent you found for this advance (from your search results)"
  },
  { ... }
]"""


# ── Strategy 1: Literature Exploration ───────────────────────────────────
# Paper: "literature exploration via web search" — search prior art AND recent advances,
# then generate inventions that are clearly distinct from or directly exploit what was found.
# Recent cross-domain advances found here act as "invention triggers".

def _literature_exploration_prompt(problem: str) -> str:
    return f"""PROBLEM: {problem}

STRATEGY: Literature Exploration

Step 1 — Search systematically. Use web_search for:
  a) Recent patents or papers directly on this problem (last 5 years)
  b) Recent breakthroughs in adjacent fields that share the same underlying physics
     (e.g. if the problem involves friction, search nanotech, biotech, food science)
  c) Any cross-domain advance that has recently made something newly possible

Step 2 — Identify what is newly enabled. For each result ask:
  "What does this allow that was not possible 5 years ago?"
  These are your invention triggers. Prioritise advances from distant domains —
  they produce the most non-obvious inventions.

Step 3 — Generate 2 inventions that each directly exploit one of these triggers.
  Each invention must name its trigger explicitly in the output fields.

{_OUTPUT_SCHEMA_LITERATURE}"""


# ── Strategy 2: Simulated Scientific Debate ───────────────────────────────
# Paper: "simulated scientific debates" — self-play between expert perspectives.
# The model argues different positions, then synthesises the strongest hybrid.

def _simulated_debate_prompt(problem: str) -> str:
    return f"""PROBLEM: {problem}

STRATEGY: Simulated Scientific Debate (Self-Play)

Simulate a structured debate between three domain experts with different backgrounds.
Run the debate in three rounds:

ROUND 1 — Opening positions (2-3 sentences each):
  Expert A (Materials Scientist): What material or surface property change solves this?
  Expert B (Process Engineer): What change to the process, sequence, or conditions solves this?
  Expert C (Systems Thinker): What change to the architecture, geometry, or integration solves this?

ROUND 2 — Critiques (1-2 sentences each):
  A critiques B's position: what is its fundamental weakness?
  B critiques C's position: what is its fundamental weakness?
  C critiques A's position: what is its fundamental weakness?

ROUND 3 — Refined positions (2 sentences each, addressing the critique received):
  Each expert refines their approach to address the critique.

SYNTHESIS — Generate 2 invention concepts from the final refined positions.
Do not simply pick one expert's view — but do not merely bundle them either.

Synthesis is only inventive if the experts' mechanisms INTERACT to produce a technical
effect none of them achieves alone. "Use A's material AND B's process AND C's geometry"
is an aggregation — each element still does its own separate job, which an examiner reads
as obvious. Instead, find where one expert's mechanism removes a limitation the others
identified in their critiques: that dependency is where a real combined effect lives.
State that effect explicitly in the summary.

If the refined positions genuinely do not interact, do not force a hybrid — take the
single strongest mechanism and deepen it instead.

{_OUTPUT_SCHEMA}"""


# ── Strategy 3: Iterative Assumptions ────────────────────────────────────
# Paper: "iterative assumptions identification" — enumerate and challenge
# implicit assumptions in the conventional approach.

def _iterative_assumptions_prompt(problem: str) -> str:
    return f"""PROBLEM: {problem}

STRATEGY: Iterative Assumptions

Step 1 — Enumerate 5 implicit assumptions in how this problem is conventionally approached.
  These are things practitioners take for granted:
  e.g. the operating temperature range, the material state (solid/liquid/gas),
       the scale (micro/macro), the process sequence, the energy source used.

Step 2 — For each assumption, ask: "What if the exact opposite were true?"
  Write one line per inversion.

Step 3 — Select the 2 most fertile inversions — those that open up the most technically
  interesting solution space. For each, develop a concrete invention that follows
  naturally from inverting the assumption.

The invention should only be possible (or natural) once the assumption is challenged.

{_OUTPUT_SCHEMA}"""


# ── Strategy 4a: Direct (Research Expansion — first principles) ───────────
# Paper: "research expansion" — takes the first-principles decomposition angle.

def _direct_prompt(problem: str) -> str:
    return f"""PROBLEM: {problem}

STRATEGY: First-Principles Research Expansion

Step 1 — Decompose the problem to its root causes at the physics or chemistry level.
  What fundamental phenomenon (friction, oxidation, diffusion, thermal gradient, etc.)
  is the limiting factor?

Step 2 — For each root cause, identify what physical law, material property,
  or thermodynamic principle could be harnessed to eliminate it.

Step 3 — Generate 2 inventions that attack the root cause directly,
  not the symptom. The mechanism should feel inevitable once the root cause is named.

{_OUTPUT_SCHEMA}"""


# ── Strategy 4b: Analogical (Research Expansion — cross-domain) ──────────
# Paper: "research expansion" via analogical reasoning from other domains.

def _analogical_prompt(problem: str) -> str:
    return f"""PROBLEM: {problem}

STRATEGY: Cross-Domain Analogical Expansion

Step 1 — Name the core physical or chemical challenge:
  (e.g. "preventing adhesion", "managing localised heat", "reducing interfacial drag")

Step 2 — Identify 3 unrelated technical domains where the exact same challenge
  appears and has been well-solved:
  (e.g. aerospace, food processing, marine biology, semiconductor fab, textile engineering)

Step 3 — Extract the specific mechanism used in each domain and adapt it
  to this problem. Name the source domain explicitly in each mechanism description.

Generate 2 inventions — each clearly derived from a different source domain.

{_OUTPUT_SCHEMA}"""


# ── Parsing helper ────────────────────────────────────────────────────────

async def _parse_inventions(
    text: str,
    strategy: str,
    session_id: str,
    model: str,
    messages: list[dict],
    search_evidence: list[str] | None = None,
) -> list[Invention]:
    raw_text = text
    for attempt in range(2):
        try:
            items = parse_json_response(raw_text)
            if not isinstance(items, list):
                raise ValueError("Expected JSON array")
            inventions = []
            for item in items[:2]:
                inventions.append(Invention(
                    session_id=session_id,
                    title=item.get("title", "Untitled Invention"),
                    summary=item.get("summary", ""),
                    mechanism=item.get("mechanism", ""),
                    strategy=strategy,
                    search_evidence=search_evidence or [],
                    # Trigger fields — only populated for literature_exploration
                    trigger_advance=item.get("trigger_advance", ""),
                    trigger_source_domain=item.get("trigger_source_domain", ""),
                    trigger_url=item.get("trigger_url", ""),
                ))
            return inventions
        except (json.JSONDecodeError, ValueError) as e:
            if attempt == 0:
                logger.warning(f"JSON parse failed for {strategy} (attempt 1): {e}")
                correction = list(messages) + [
                    {"role": "assistant", "content": raw_text},
                    {"role": "user", "content": (
                        "Your response was not valid JSON. Return ONLY the JSON array "
                        f"with exactly 2 invention objects. No other text.{_OUTPUT_SCHEMA}"
                    )},
                ]
                try:
                    raw_text, _ = await run_claude_with_tools(
                        model=model, system=_BASE_SYSTEM,
                        messages=correction, tools=[], max_tool_rounds=1,
                    )
                except Exception:
                    pass
            else:
                logger.error(f"Generation parse failed for {strategy}: {e}")
    return []


# ── Individual strategy runner ────────────────────────────────────────────

async def _run_strategy(
    strategy: str,
    user_prompt: str,
    session_id: str,
    on_progress: Callable,
    use_search: bool = False,
) -> list[Invention]:
    tools = [WEB_SEARCH_TOOL] if use_search else []
    messages = [{"role": "user", "content": user_prompt}]

    async def _on_search(agent: str, query: str, results: list) -> None:
        await on_progress("search_query", {
            "agent": agent,
            "query": query,
            "result_count": len(results),
            "top_results": [
                {"title": r.get("title", ""), "url": r.get("url", "")}
                for r in results[:3]
            ],
        })

    try:
        text, search_results = await run_claude_with_tools(
            model=settings.r_generation_model,
            system=_BASE_SYSTEM,
            messages=messages,
            tools=tools,
            on_tool_call=_on_search if use_search else None,
            agent_name=f"generation/{strategy}",
            max_tool_rounds=6 if use_search else 1,
        )
    except AgentCallError as e:
        logger.error(f"Strategy {strategy} failed: {e}")
        return []

    search_urls = [r.get("url", "") for r in search_results if r.get("url")]
    return await _parse_inventions(text, strategy, session_id,
                                   settings.r_generation_model, messages, search_urls)


# ── Prior round context block ─────────────────────────────────────────────

def _build_prior_context_block(prior_context: dict) -> str:
    """
    Formats context from a previous round into a preamble injected into every
    generation strategy prompt. Tells the model what was already explored so
    it generates genuinely new ideas.
    """
    round_n = prior_context.get("round_number", 1)
    top_inventions = prior_context.get("top_inventions", [])
    meta = prior_context.get("meta_review", {})
    user_feedback = prior_context.get("user_feedback", "")

    lines = [
        f"═══ CONTEXT FROM ROUND {round_n} ═══",
        "",
        "These mechanisms were already explored — do NOT repeat or closely echo them:",
    ]
    for inv in top_inventions[:5]:
        lines.append(f"  • {inv.get('title','')}: {inv.get('mechanism','')[:120]}")

    if meta.get("recurring_challenges"):
        lines.append("")
        lines.append("Known recurring challenges (your ideas should address these):")
        for c in meta["recurring_challenges"][:3]:
            lines.append(f"  • {c[:120]}")

    if meta.get("unexplored_directions"):
        lines.append("")
        lines.append("Unexplored directions identified (fertile ground for new ideas):")
        for d in meta["unexplored_directions"][:3]:
            lines.append(f"  • {d[:120]}")

    if user_feedback:
        lines.append("")
        lines.append(f"USER STEERING FOR THIS ROUND: {user_feedback}")

    lines.append("")
    lines.append("Generate inventions that go BEYOND everything listed above.")
    lines.append("═══════════════════════════════════")
    lines.append("")

    return "\n".join(lines)


# ── Main entry point ──────────────────────────────────────────────────────

async def run(
    problem_statement: str,
    session_id: str,
    on_progress: Callable,
    prior_context: dict | None = None,
) -> GenerationResult:
    """
    Run all generation strategies concurrently.

    prior_context — if provided (round > 1), contains:
      round_number, top_inventions, meta_review, user_feedback
    Each strategy prompt is prefixed with this context so the model
    generates novel ideas rather than repeating what was already found.
    """
    # Build context preamble for round > 1
    ctx_block = _build_prior_context_block(prior_context) if prior_context else ""

    def with_context(prompt: str) -> str:
        return ctx_block + prompt if ctx_block else prompt

    tasks = [
        _run_strategy("literature_exploration",
                      with_context(_literature_exploration_prompt(problem_statement)),
                      session_id, on_progress, use_search=True),
        _run_strategy("simulated_debate",
                      with_context(_simulated_debate_prompt(problem_statement)),
                      session_id, on_progress, use_search=False),
        _run_strategy("iterative_assumptions",
                      with_context(_iterative_assumptions_prompt(problem_statement)),
                      session_id, on_progress, use_search=False),
        _run_strategy("direct",
                      with_context(_direct_prompt(problem_statement)),
                      session_id, on_progress, use_search=False),
        _run_strategy("analogical",
                      with_context(_analogical_prompt(problem_statement)),
                      session_id, on_progress, use_search=False),
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    inventions: list[Invention] = []
    for r in results:
        if isinstance(r, Exception):
            logger.error(f"Generation strategy error: {r}")
            continue
        inventions.extend(r)

    for inv in inventions:
        await on_progress("invention_generated", {
            "invention_id": inv.id,
            "title": inv.title,
            "strategy": inv.strategy,
        })

    logger.info(f"Generation: {len(inventions)} inventions from {len(tasks)} strategies")
    return GenerationResult(inventions=inventions)
