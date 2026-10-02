# rag/evidence.py — Turn raw record fields into EvidenceState facts.
# Shared by rag/filter.py (verified-only hard filter) and rag/rank.py (scoring).
# Rule: anything not explicitly verified is UNKNOWN, never VERIFIED_NO.

import re
from dataclasses import dataclass, field

from rag.citations import _hash8
from rag.config import AFFORDABILITY_SCORES
from rag.schemas import EvidenceState

# program_type text (VERIFIED_ACCESS_PROGRAMS) -> tier. Checked in order.
_PROGRAM_TIERS = [
    ("no_assistance", r"\b(no assistance|no sliding|not offered|none)\b"),
    ("sliding_fee", r"\bsliding[- ]?(fee|scale)\b"),
    ("free", r"\b(free|no[- ]cost|charity)\b"),
    ("discounted_self_pay", r"\b(discount\w*|self[- ]pay)\b"),
    ("cash_price_only", r"\bcash price\b"),
]

_YES = {"yes", "y", "true", "1", "verified_yes"}
_NO = {"no", "n", "false", "0", "verified_no"}


@dataclass
class AffordabilityEvidence:
    state: EvidenceState
    tier: str
    score: float
    source_ids: list[str] = field(default_factory=list)
    description: str = ""


def program_source_id(record: dict) -> str:
    return f"VERIFIED_PROGRAM:{record.get('npi') or 'unknown'}:{_hash8(record.get('program_type'), record.get('verification_url'))}"


def affordability_evidence(record: dict) -> AffordabilityEvidence:
    """Verified only when a curated program row is attached (program_type + verification_url/verified_at).
    NPPES' 'Not stated in NPPES — call to verify' is UNKNOWN."""
    program = str(record.get("program_type") or "").strip()
    verified = bool(record.get("verification_url") or record.get("verified_at"))
    if program and verified:
        low = program.lower()
        for tier, pattern in _PROGRAM_TIERS:
            if re.search(pattern, low):
                state = EvidenceState.VERIFIED_NO if tier == "no_assistance" else EvidenceState.VERIFIED_YES
                return AffordabilityEvidence(state, tier, AFFORDABILITY_SCORES[tier],
                                             [program_source_id(record)], f"Verified program: {program}")
    return AffordabilityEvidence(EvidenceState.UNKNOWN, "unknown", AFFORDABILITY_SCORES["unknown"], [],
                                 "Affordability not stated in source data")


def availability_evidence(record: dict) -> tuple[EvidenceState, str]:
    """Only an explicitly verified flag counts. Missing or unverified -> UNKNOWN."""
    for key, label in (("accepting_new_patients", "Accepting new patients"),
                       ("walk_in", "Walk-in service"),
                       ("hours_verified", "Current hours")):
        val = str(record.get(key) or "").strip().lower()
        if not val or not record.get("availability_source"):
            continue
        if val in _YES:
            return EvidenceState.VERIFIED_YES, f"{label}: verified"
        if val in _NO:
            return EvidenceState.VERIFIED_NO, f"{label}: verified not offered"
    return EvidenceState.UNKNOWN, "Availability (new patients, walk-in, hours) not verified"


def is_active(record: dict) -> EvidenceState:
    """TEMPE_PROVIDERS is built WHERE IS_ACTIVE = TRUE, so NPPES rows without an
    explicit flag are verified active by construction."""
    val = str(record.get("active", record.get("is_active", "")) or "").strip().lower()
    if val in _NO:
        return EvidenceState.VERIFIED_NO
    if val in _YES or "nppes" in str(record.get("source") or "").lower():
        return EvidenceState.VERIFIED_YES
    return EvidenceState.UNKNOWN
