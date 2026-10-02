# rag/schemas.py — Shared data models for the Top-3 ranking pipeline.
#
# Unknown-sensitive facts use EvidenceState instead of bool so that
# "not known" is never confused with "verified no".

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

Rank = Literal["A", "B", "C"]
RANK_LABELS: tuple[Rank, ...] = ("A", "B", "C")


class EvidenceState(str, Enum):
    VERIFIED_YES = "verified_yes"
    VERIFIED_NO = "verified_no"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class ScoredFact(BaseModel):
    """One ranking dimension with the evidence that produced its score."""
    name: str
    score: float | None
    explanation: str
    source_ids: list[str] = Field(default_factory=list)
    evidence_state: EvidenceState


class RankingBreakdown(BaseModel):
    service_match: float
    proximity: float | None
    financial: float | None
    confidence: float
    availability: float | None
    total: float

    distance_miles: float | None = None
    distance_method: str | None = None

    financial_evidence_state: EvidenceState = EvidenceState.UNKNOWN
    availability_evidence_state: EvidenceState = EvidenceState.UNKNOWN


class RankedProvider(BaseModel):
    rank: Rank | None = None
    source_id: str
    npi: str
    name: str
    category: str
    specialty: str
    address: str
    phone: str | None

    scores: RankingBreakdown
    facts: list[ScoredFact] = Field(default_factory=list)
    evidence_facts: list[str] = Field(default_factory=list)
    unknown_facts: list[str] = Field(default_factory=list)


class ProviderExplanation(BaseModel):
    """LLM (or template) explanation for one already-ranked provider."""
    rank: Rank
    source_id: str
    heading: str
    reason: str
    proximity_explanation: str
    financial_explanation: str
    service_explanation: str
    important_unknowns: list[str]
    comparison_to_next: str | None
    citation_source_ids: list[str]


class TopThreeExplanation(BaseModel):
    query_interpretation: str
    providers: list[ProviderExplanation]
    ranking_disclaimer: str


class TopProviderResponse(BaseModel):
    query_understood_as: str
    ranking_basis: list[str]
    providers: list[RankedProvider]
    explanations: list[ProviderExplanation] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    emergency: bool = False
    eliminated_by: dict[str, int] = Field(default_factory=dict)
