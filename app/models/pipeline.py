from __future__ import annotations

from pydantic import BaseModel

from app.models.invention import Invention, Review


class GenerationResult(BaseModel):
    inventions: list[Invention]


class ProximityResult(BaseModel):
    inventions: list[Invention]     # Deduplicated list
    removed_ids: list[str]
    clusters: list[dict]            # [{label, theme, invention_ids}]


class ReflectionResult(BaseModel):
    reviews: list[Review]


class RankingResult(BaseModel):
    ranked_inventions: list[Invention]


class EvolutionResult(BaseModel):
    evolved_inventions: list[Invention]
    final_ranked: list[Invention]


class MetaReviewResult(BaseModel):
    overview: str
    strongest_approaches: list[str]
    recurring_challenges: list[str]
    unexplored_directions: list[str]
    cross_domain_insight: str
    recommendation: str
