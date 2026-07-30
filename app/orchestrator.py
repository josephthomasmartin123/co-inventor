"""
Pipeline Orchestrator — Co-Scientist paper-faithful implementation

Pipeline order (matches paper):
  1. Generation  — all 5 strategies concurrently (literature exploration naturally
                   finds recent cross-domain advances / "triggers")
  2. Proximity   — deduplicate near-identical inventions, cluster by approach
  3. Reflection  — two-tier: initial filter → full web-search evaluation
  4. Ranking     — Elo tournament, pairwise scientific debates; everything enters level
  5. Evolution   — enhance top-K, combine across mechanistic families only, then
                   (5b) review the variants and re-run the tournament, because the paper
                   requires every new hypothesis to compete rather than be promoted
  6. Meta-review — synthesise all evaluations into a research overview

Deliberate departures from the paper are documented in docs/ADAPTATION.md.

Progress events emitted to asyncio.Queue, consumed by SSE endpoint.
All search queries surface as 'search_query' events so the UI can show them.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime

from app.agents import evolution, generation, meta_review, proximity, ranking, reflection
from app.models.session import ProgressEvent, Session
from app.storage import Storage

logger = logging.getLogger(__name__)


async def run_pipeline(
    session: Session,
    storage: Storage,
    event_queue: asyncio.Queue,
) -> None:
    sid = session.id

    async def emit(event_type: str, data: dict) -> None:
        event = ProgressEvent(event=event_type, data=data, session_id=sid)
        await event_queue.put(event.model_dump())

    # on_progress is the universal callback passed into every agent.
    # Agents use it to emit: invention_generated, search_query, review_complete, etc.
    async def on_progress(event_type: str, data: dict) -> None:
        await emit(event_type, data)

    try:
        # ── Stage 1: Generation ───────────────────────────────────────────
        await storage.update_session_status(sid, "generating")
        await emit("status", {
            "stage": "generating",
            "stage_num": 1, "total_stages": 6,
            "message": "Running 5 generation strategies in parallel…",
            "detail": "literature_exploration · simulated_debate · iterative_assumptions · direct · analogical",
        })

        gen_result = await generation.run(
            problem_statement=session.problem_statement,
            session_id=sid,
            on_progress=on_progress,
            prior_context=session.prior_context if hasattr(session, "prior_context") else None,
        )
        await storage.save_inventions(gen_result.inventions)

        # ── Stage 2: Proximity ────────────────────────────────────────────
        await storage.update_session_status(sid, "proximity")
        await emit("status", {
            "stage": "proximity",
            "stage_num": 2, "total_stages": 6,
            "message": f"Deduplicating {len(gen_result.inventions)} inventions and clustering by approach…",
        })

        prox_result = await proximity.run(
            inventions=gen_result.inventions,
            session_id=sid,
            on_progress=on_progress,
        )
        # Save removed inventions too (they'll still appear in DB, just not ranked)
        await storage.save_inventions(prox_result.inventions)

        # ── Stage 3: Reflection ───────────────────────────────────────────
        await storage.update_session_status(sid, "reflecting")
        n_inv = len(prox_result.inventions)
        await emit("status", {
            "stage": "reflecting",
            "stage_num": 3, "total_stages": 6,
            "message": f"Two-tier evaluation of {n_inv} inventions: initial filter → full prior art search…",
        })

        refl_result = await reflection.run(
            inventions=prox_result.inventions,
            problem_statement=session.problem_statement,
            session_id=sid,
            on_progress=on_progress,
        )
        reviews_by_id = {r.invention_id: r for r in refl_result.reviews}
        await storage.save_reviews(refl_result.reviews)

        # ── Stage 4: Ranking ──────────────────────────────────────────────
        await storage.update_session_status(sid, "ranking")
        await emit("status", {
            "stage": "ranking",
            "stage_num": 4, "total_stages": 6,
            "message": f"Elo tournament — {settings_rounds()} pairwise scientific debates…",
        })

        rank_result = await ranking.run(
            inventions=prox_result.inventions,
            reviews=reviews_by_id,
            problem_statement=session.problem_statement,
            session_id=sid,
            on_progress=on_progress,
            clusters=prox_result.clusters,
        )
        await storage.update_inventions(rank_result.ranked_inventions)

        # ── Stage 5: Evolution ────────────────────────────────────────────
        await storage.update_session_status(sid, "evolving")
        await emit("status", {
            "stage": "evolving",
            "stage_num": 5, "total_stages": 6,
            "message": "Refining top inventions — enhancing specificity and combining best concepts…",
        })

        evo_result = await evolution.run(
            ranked_inventions=rank_result.ranked_inventions,
            reviews=reviews_by_id,
            problem_statement=session.problem_statement,
            session_id=sid,
            on_progress=on_progress,
            # Evolution pairs on mechanistic family, not rank — two top-ranked
            # inventions from one family are variants, not complements.
            clusters=prox_result.clusters,
        )
        await storage.save_inventions(evo_result.evolved_inventions)

        # ── Stage 5b: Variants must earn their place ───────────────────────
        # The paper does not let evolution promote anything on its own: "The Evolution
        # agent generates new hypotheses; it doesn't modify or replace existing ones.
        # This strategy protects the quality of top-ranked hypotheses from flawed
        # improvements, as each new hypothesis must also compete in the tournament."
        # So a variant is reviewed like any other invention, then enters the tournament
        # at the same rating as everything else and has to win to place above its parent.
        final_ranked = rank_result.ranked_inventions

        if evo_result.evolved_inventions:
            n_new = len(evo_result.evolved_inventions)
            await emit("status", {
                "stage": "evolving",
                "stage_num": 5, "total_stages": 6,
                "message": f"Reviewing {n_new} new variant{'s' if n_new != 1 else ''} "
                           f"against the same five dimensions…",
            })

            evo_refl = await reflection.run(
                inventions=evo_result.evolved_inventions,
                problem_statement=session.problem_statement,
                session_id=sid,
                on_progress=on_progress,
            )
            reviews_by_id.update({r.invention_id: r for r in evo_refl.reviews})
            await storage.save_reviews(evo_refl.reviews)

            await emit("status", {
                "stage": "evolving",
                "stage_num": 5, "total_stages": 6,
                "message": "Re-running the tournament — variants must beat the "
                           "inventions they came from to rank above them…",
            })

            final_rank_result = await ranking.run(
                inventions=evo_result.final_ranked,
                reviews=reviews_by_id,
                problem_statement=session.problem_statement,
                session_id=sid,
                on_progress=on_progress,
                clusters=prox_result.clusters,
                # Newcomers are prioritised for matches, so they are actually tested
                # rather than coasting on their entry rating.
                new_ids={inv.id for inv in evo_result.evolved_inventions},
                phase="re-run",
            )
            final_ranked = final_rank_result.ranked_inventions
            await storage.update_inventions(final_ranked)

        # ── Stage 6: Meta-review ──────────────────────────────────────────
        await storage.update_session_status(sid, "meta_reviewing")
        await emit("status", {
            "stage": "meta_reviewing",
            "stage_num": 6, "total_stages": 6,
            "message": "Synthesising all evaluations into a research overview…",
        })

        meta_result = await meta_review.run(
            problem_statement=session.problem_statement,
            ranked_inventions=final_ranked,
            reviews=reviews_by_id,
            session_id=sid,
            on_progress=on_progress,
        )

        # ── Finalise ──────────────────────────────────────────────────────
        final_top = final_ranked[:5]
        final_ids = [inv.id for inv in final_top]
        meta_dict = meta_result.model_dump()

        await storage.finalize_session(sid, final_ids, meta_dict)

        top = final_top[0] if final_top else None
        await emit("done", {
            "final_ranked_ids": final_ids,
            "total_inventions_generated": len(gen_result.inventions),
            "after_dedup": len(prox_result.inventions),
            "top_invention": {
                "title": top.title if top else "",
                "strategy": top.strategy if top else "",
                "elo_score": round(top.elo_score, 0) if top else 0,
            },
            "meta_review": meta_dict,
        })

        logger.info(f"Pipeline complete for session {sid}: {len(final_top)} top inventions")

    except Exception as e:
        error_msg = str(e)
        logger.exception(f"Pipeline failed for {sid}: {error_msg}")
        await storage.update_session_status(sid, "failed", error=error_msg)
        await emit("error", {"message": error_msg})


def settings_rounds() -> int:
    from app.config import settings
    return settings.elo_rounds
