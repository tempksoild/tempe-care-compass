# rag/validate.py — Check an explanation against the code-selected providers (task 8).
#
# Structural: 0-3 providers, ranks are an A,B,C prefix, same source IDs in the same
# order, citations are a subset of the provider's supplied (public) source IDs.
# Claims: free/low-cost, walk-in/new patients/hours, miles, "expensive", and quality
# claims are only allowed when the provider's evidence verifies them. A clause that
# negates or hedges ("not verified", "unknown", "call to confirm") is not a claim.
# On any violation agent._explain_ranking falls back to the deterministic template.

import re

from rag.explain import labelled, public_source_ids
from rag.schemas import EvidenceState, RankedProvider, TopThreeExplanation

HEDGE = re.compile(r"\b(not|unknown|unverified|isn'?t|aren'?t|wasn'?t|cannot|can'?t|unavailable|"
                   r"confirm|verify|whether|no information|n/a)\b", re.I)
CLAUSE_SPLIT = re.compile(r"[.;!?\n]|\bbut\b|\bhowever\b|\balthough\b", re.I)

AFFORDABILITY_CLAIM = re.compile(r"\b(free|no[- ]cost|low[- ]cost|sliding[- ]?(fee|scale)|charity care|"
                                 r"discount(ed)?|affordable|cheap(er|est)?|inexpensive)\b", re.I)
AVAILABILITY_CLAIM = re.compile(r"\b(accept(s|ing)? new patients|walk[- ]?ins?|open (now|today|late|24)|"
                                r"same[- ]day|currently open|hours (are|of))\b", re.I)
EXPENSIVE_CLAIM = re.compile(r"\b(expensive|costly|overpriced|pricier|higher[- ]priced)\b", re.I)
QUALITY_CLAIM = re.compile(r"\b(best (provider|doctor|clinic|care|quality)|highest[- ]quality|top[- ]rated|"
                           r"most qualified|better (doctor|quality|care)|is the best)\b", re.I)
MILES = re.compile(r"(\d+(?:\.\d+)?)\s*(?:mi\b|miles?\b)", re.I)


class ExplanationValidationError(ValueError):
    def __init__(self, violations: list[str]):
        super().__init__("; ".join(violations))
        self.violations = violations


def _clauses(text: str | None):
    for c in CLAUSE_SPLIT.split(text or ""):
        c = c.strip()
        if c:
            yield c


def _claims(text: str, pattern: re.Pattern) -> list[str]:
    return [c for c in _clauses(text) if pattern.search(c) and not HEDGE.search(c)]


def _provider_text_fields(e) -> dict[str, str]:
    return {"heading": e.heading, "reason": e.reason, "proximity_explanation": e.proximity_explanation,
            "financial_explanation": e.financial_explanation, "service_explanation": e.service_explanation,
            "comparison_to_next": e.comparison_to_next or ""}


def check_claims(label: str, e, p: RankedProvider, nxt: RankedProvider | None) -> list[str]:
    out = []
    aff_ok = p.scores.financial_evidence_state == EvidenceState.VERIFIED_YES and p.scores.financial_basis in (
        "affordability", "price")
    avail_ok = p.scores.availability_evidence_state == EvidenceState.VERIFIED_YES
    price_known = p.scores.price_score is not None
    allowed_miles = {round(x.scores.distance_miles, 1) for x in (p, nxt) if x and x.scores.distance_miles is not None}

    for field, text in _provider_text_fields(e).items():
        where = f"{label}.{field}"
        if not aff_ok:
            out += [f"{where}: unverified affordability claim '{c}'" for c in _claims(text, AFFORDABILITY_CLAIM)]
        if not avail_ok:
            out += [f"{where}: unverified availability claim '{c}'" for c in _claims(text, AVAILABILITY_CLAIM)]
        if not price_known:
            out += [f"{where}: price called expensive without a price '{c}'" for c in _claims(text, EXPENSIVE_CLAIM)]
        out += [f"{where}: quality claim '{c}'" for c in _claims(text, QUALITY_CLAIM)]
        for m in MILES.finditer(text):
            if round(float(m.group(1)), 1) not in allowed_miles:
                out.append(f"{where}: distance '{m.group(0)}' not supported by a calculated distance")
    return out


def validate(explanation: TopThreeExplanation, selected: list[RankedProvider]) -> list[str]:
    """Return a list of violations (empty = valid)."""
    v = []
    pairs = labelled(selected)
    n = len(explanation.providers)
    if not 0 <= n <= 3:
        v.append(f"expected 0-3 providers, got {n}")
    if selected and n == 0:
        v.append("explanation has no providers but providers were selected")
    expected = [(label, p.source_id) for label, p in pairs]
    got = [(e.rank, e.source_id) for e in explanation.providers]
    if [r for r, _ in got] != ["A", "B", "C"][:n]:
        v.append(f"ranks must be an A,B,C prefix, got {[r for r, _ in got]}")
    if {s for _, s in got} != {s for _, s in expected}:
        v.append("source-ID set differs from the selected providers")
    elif got != expected:
        v.append("providers were reordered")
    if v:
        return v  # claim checks need a 1:1 mapping

    for i, (e, (label, p)) in enumerate(zip(explanation.providers, pairs)):
        allowed = set(public_source_ids(p))
        if not e.citation_source_ids:
            v.append(f"{label}: no citations")
        extra = [s for s in e.citation_source_ids if s not in allowed]
        if extra:
            v.append(f"{label}: citations not in supplied IDs {extra}")
        nxt = pairs[i + 1][1] if i + 1 < len(pairs) else None
        v += check_claims(label, e, p, nxt)

    for field in ("query_interpretation", "ranking_disclaimer"):
        text = getattr(explanation, field)
        v += [f"{field}: quality claim '{c}'" for c in _claims(text, QUALITY_CLAIM)]
        if field == "ranking_disclaimer":
            v += [f"{field}: availability claim '{c}'" for c in _claims(text, AVAILABILITY_CLAIM)]
    return v


def validate_explanation(explanation: TopThreeExplanation, selected: list[RankedProvider]) -> TopThreeExplanation:
    violations = validate(explanation, selected)
    if violations:
        raise ExplanationValidationError(violations)
    return explanation
