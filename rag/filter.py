# rag/filter.py — Eligibility filtering (hard requirements). Ranking happens later.
#
# Each candidate is removed by the FIRST constraint it fails; counts are kept per
# constraint so a zero-result answer can say what eliminated the candidates.
# Order: invalid -> inactive -> outside_area -> wrong_category -> missing_procedure
#        -> not_verified_affordable -> duplicate
# Duplicates are checked last so only otherwise-eligible rows compete; the first
# one in retrieval (RRF) order is kept.

import re

from pydantic import BaseModel, Field

from rag.citations import normalized_location
from rag.config import SERVICE_AREA
from rag.evidence import affordability_evidence, is_active
from rag.intent import RetrievalIntent
from rag.retrieve import Candidate
from rag.schemas import EvidenceState

CONSTRAINT_LABELS = {
    "invalid_record": "invalid or incomplete record (NPI/name)",
    "inactive": "inactive provider",
    "outside_area": "outside the requested area",
    "wrong_category": "different care category than requested",
    "missing_procedure": "no matching billing procedure",
    "not_verified_affordable": "no verified affordability program",
    "duplicate": "duplicate NPI + location",
}


class EligibilityResult(BaseModel):
    eligible: list[Candidate] = Field(default_factory=list)
    eliminated_by: dict[str, int] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    considered: int = 0


def _invalid(r: dict) -> bool:
    return not re.fullmatch(r"\d{10}", str(r.get("npi") or "").strip()) or not str(r.get("name") or "").strip()


def _outside_area(r: dict, intent: RetrievalIntent, require_zip: bool) -> bool:
    if str(r.get("state") or "").strip().upper() != SERVICE_AREA["state"]:
        return True
    if str(r.get("city") or "").strip().lower() != SERVICE_AREA["city"]:
        return True
    if require_zip and intent.zip_code:
        return str(r.get("zip") or "")[:5] != intent.zip_code
    return False


class EligibilityFilter:
    """require_zip: treat the requested ZIP as a hard area limit (default: ranking factor only).
    price_index: {provider source_id -> {(code_type, code), ...}}. Without it there is no
    provider-to-price linkage, so the procedure filter is skipped and noted as a limitation."""

    def __init__(self, require_zip: bool = False, price_index: dict[str, set] | None = None):
        self.require_zip = require_zip
        self.price_index = price_index

    def _failed(self, c: Candidate, intent: RetrievalIntent) -> str | None:
        r = c.record
        if _invalid(r):
            return "invalid_record"
        if is_active(r) == EvidenceState.VERIFIED_NO:
            return "inactive"
        if _outside_area(r, intent, self.require_zip):
            return "outside_area"
        if intent.category != "All" and r.get("category") != intent.category:
            return "wrong_category"
        if intent.price_query and intent.billing_codes and self.price_index is not None:
            codes = self.price_index.get(c.source_id, set())
            if not codes & set(map(tuple, intent.billing_codes)):
                return "missing_procedure"
        if intent.verified_only and affordability_evidence(r).state != EvidenceState.VERIFIED_YES:
            return "not_verified_affordable"
        return None

    def apply(self, candidates: list[Candidate], intent: RetrievalIntent) -> EligibilityResult:
        res = EligibilityResult(considered=len(candidates))
        seen: set[tuple[str, str]] = set()
        for c in candidates:
            reason = self._failed(c, intent)
            if reason is None:
                key = (str(c.record.get("npi")).strip(), normalized_location(c.record))
                if key in seen:
                    reason = "duplicate"
                else:
                    seen.add(key)
            if reason:
                res.eliminated_by[reason] = res.eliminated_by.get(reason, 0) + 1
            else:
                res.eligible.append(c)
        if intent.price_query and intent.billing_codes and self.price_index is None:
            res.limitations.append("Provider records are not linked to price data, so the procedure filter was not applied.")
        return res

    @staticmethod
    def describe(eliminated_by: dict[str, int]) -> list[str]:
        return [f"{n} removed: {CONSTRAINT_LABELS.get(k, k)}"
                for k, n in sorted(eliminated_by.items(), key=lambda kv: (-kv[1], kv[0]))]
