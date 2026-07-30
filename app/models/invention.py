from __future__ import annotations

import uuid
from typing import Optional

from pydantic import BaseModel, Field


class Trigger(BaseModel):
    """
    An invention trigger: a recent advance or breakthrough that enables new inventions.

    The Co-Inventor searches for triggers before generating ideas. The best triggers
    come from adjacent or distant domains — like how nanobubble research from food
    science triggered an invention for concrete workability.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    advance: str                    # What the advance/discovery is
    source_domain: str              # Domain where this advance originated
    mechanism: str                  # How it works / what it enables physically/chemically
    relevance: str                  # Why this might help solve the stated problem
    source_url: Optional[str] = ""  # URL of paper/article describing the advance
    recency: str = ""               # Approximate date/year of the advance if known
    distance: str = "adjacent"      # "same_domain" | "adjacent" | "distant"


class Invention(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    title: str                          # Short noun-phrase title
    summary: str                        # 2-3 sentence description
    mechanism: str                      # How it works technically
    strategy: str                       # See STRATEGY_DESCRIPTIONS in frontend for full list
    search_evidence: list[str] = []     # URLs found during generation search
    # Paper: "We set the initial Elo rating of 1200 for the newly added hypothesis."
    # Every invention enters the tournament here, whenever it is created — an evolved
    # variant gets no head start over the concept it came from. Only match outcomes
    # move a rating.
    elo_score: float = 1200.0
    elo_wins: int = 0
    elo_losses: int = 0

    # Trigger fields — populated for literature_exploration inventions only.
    # Captures the specific recent advance (cross-domain or otherwise) that
    # made this invention newly conceivable.
    trigger_advance: str = ""           # What the advance is, in one sentence
    trigger_source_domain: str = ""     # Domain it came from (e.g. "food science")
    trigger_url: str = ""              # Source URL found during search


class Review(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    invention_id: str
    session_id: str
    # Five dimensions — matches the paper's evaluation criteria adapted for invention:
    # novelty, scientific plausibility, feasibility, problem fit, patentability
    novelty_score: int                          # 1-5  (non-obvious mechanism, absent from prior art)
    novelty_rationale: str
    prior_art_found: list[str] = []             # URLs of potentially conflicting prior art
    scientific_plausibility_score: int          # 1-5  (underlying physics/chemistry is sound)
    scientific_plausibility_rationale: str
    feasibility_score: int                      # 1-5  (can be built with current/near-future tech)
    feasibility_rationale: str
    problem_fit_score: int                      # 1-5  (addresses the root cause, not a symptom)
    problem_fit_rationale: str
    patentability_score: int                    # 1-5  (specific, claimable technical mechanism)
    patentability_rationale: str
    overall_score: float                        # Weighted average


class EloMatchup(BaseModel):
    invention_a_id: str
    invention_b_id: str
    winner_id: str
    rationale: str
    round: int
