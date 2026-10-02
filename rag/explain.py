# rag/explain.py — Turn already-ranked providers into A/B/C explanations.
#
# Code prepares everything the model may say (scores, facts, source IDs, unknowns,
# score gaps to the next provider). The model only words it. Used by
# agent.CareAgent / agent.CortexAgent .explain_ranking(); template_explanation()
# is the deterministic fallback when no LLM is available or its output is unusable.

import json

from rag.intent import RetrievalIntent
from rag.schemas import RANK_LABELS, ProviderExplanation, RankedProvider, TopThreeExplanation

HEADINGS = {"A": "Best overall match", "B": "Strong alternative", "C": "Third option"}
DIMENSIONS = ("service", "proximity", "financial", "confidence", "availability")
DIM_LABEL = {"service": "service match", "proximity": "location match", "financial": "price/affordability",
             "confidence": "data confidence", "availability": "verified availability"}
DISCLAIMER = ("A is the highest-ranked match in the current dataset for the factors in this request, "
              "not a measure of clinical quality. Call to confirm price, eligibility, hours and new-patient status.")

GENERATOR_PROMPT = """You are explaining an already-calculated provider ranking.

Do not select, reorder, remove or add providers.
A, B and C were ranked by deterministic application logic.
Use only the supplied ranking facts.
Describe A as the highest-ranked match in the current dataset,
not as the highest-quality medical provider.

For every provider:
1. Explain service match.
2. Explain proximity, but only if distance is verified. A ZIP or city match is not a distance; never state miles unless distance_miles is supplied.
3. Explain price or affordability.
4. State important unknowns.
5. Cite only supplied source IDs (in citation_source_ids).
Use comparison_to_next to explain why this provider ranks above the next one, using score_gap_to_next. Use an empty string for the last provider.

Never infer affordability from provider category.
Never claim an unknown price is expensive.
Never claim a provider is accepting patients, walk-in, open,
free or low-cost unless supplied evidence explicitly verifies it.
Return JSON matching the schema exactly."""

# Hand-written (flat, no $refs, all required) so it works for both Cortex
# Messages output_config and Ollama `format`.
EXPLANATION_SCHEMA = {
    "type": "object",
    "properties": {
        "query_interpretation": {"type": "string"},
        "providers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "rank": {"type": "string", "enum": list(RANK_LABELS)},
                    "source_id": {"type": "string"},
                    "heading": {"type": "string"},
                    "reason": {"type": "string"},
                    "proximity_explanation": {"type": "string"},
                    "financial_explanation": {"type": "string"},
                    "service_explanation": {"type": "string"},
                    "important_unknowns": {"type": "array", "items": {"type": "string"}},
                    "comparison_to_next": {"type": "string"},
                    "citation_source_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["rank", "source_id", "heading", "reason", "proximity_explanation",
                             "financial_explanation", "service_explanation", "important_unknowns",
                             "comparison_to_next", "citation_source_ids"],
                "additionalProperties": False,
            },
        },
        "ranking_disclaimer": {"type": "string"},
    },
    "required": ["query_interpretation", "providers", "ranking_disclaimer"],
    "additionalProperties": False,
}


def labelled(providers: list[RankedProvider]) -> list[tuple[str, RankedProvider]]:
    if len(providers) > len(RANK_LABELS):
        raise ValueError(f"explain at most {len(RANK_LABELS)} providers; got {len(providers)} (select first)")
    return [(p.rank or RANK_LABELS[i], p) for i, p in enumerate(providers)]


def _fact(p: RankedProvider, name: str):
    return next((f for f in p.facts if f.name == name), None)


def _dim_scores(p: RankedProvider) -> dict[str, float | None]:
    s = p.scores
    return {"service": s.service_match, "proximity": s.proximity, "financial": s.financial,
            "confidence": s.confidence, "availability": s.availability}


def public_source_ids(p: RankedProvider) -> list[str]:
    """Citable IDs. USER_LOCATION:* is internal (origin of a location calc), not a citation."""
    ids = [p.source_id] + [sid for f in p.facts for sid in f.source_ids]
    return [i for i in dict.fromkeys(ids) if not i.startswith("USER_LOCATION:")]


def score_gap(p: RankedProvider, nxt: RankedProvider | None) -> dict | None:
    if nxt is None:
        return None
    a, b = _dim_scores(p), _dim_scores(nxt)
    gaps = {d: round(a[d] - b[d], 2) for d in DIMENSIONS if a[d] is not None and b[d] is not None and a[d] != b[d]}
    return {"total": round(p.scores.total - nxt.scores.total, 2), "by_dimension": gaps,
            "decided_by_tie_break": p.scores.total == nxt.scores.total}


def describe_intent(intent: RetrievalIntent) -> str:
    parts = [intent.category if intent.category != "All" else "Any care type"]
    if intent.zip_code:
        parts.append(f"near ZIP {intent.zip_code}")
    if intent.keywords:
        parts.append("keywords: " + ", ".join(intent.keywords))
    if intent.price_query:
        parts.append(f"price question{' about ' + intent.procedure if intent.procedure else ''}")
    if intent.verified_only:
        parts.append("verified affordable programs only")
    elif intent.affordability_required:
        parts.append("affordability matters")
    return "; ".join(parts)


def prepare_records(providers: list[RankedProvider]) -> list[dict]:
    """The only provider data the LLM sees: the selected (≤3) providers."""
    pairs = labelled(providers)
    out = []
    for i, (label, p) in enumerate(pairs):
        nxt = pairs[i + 1] if i + 1 < len(pairs) else None
        reasons = {}
        for d in DIMENSIONS:
            f = _fact(p, d)
            if f is None:
                continue
            reasons[d] = {"score": f.score, "fact": f.explanation, "evidence_state": f.evidence_state.value,
                          "source_ids": [s for s in f.source_ids if not s.startswith("USER_LOCATION:")]}
        out.append({
            "rank": label, "source_id": p.source_id, "name": p.name, "category": p.category,
            "specialty": p.specialty, "total_score": p.scores.total,
            "distance_miles": p.scores.distance_miles, "distance_method": p.scores.distance_method,
            "ranking_reasons": reasons, "unknowns": p.unknown_facts,
            "next_rank": nxt[0] if nxt else None,
            "score_gap_to_next": score_gap(p, nxt[1] if nxt else None),
            "allowed_source_ids": public_source_ids(p),
        })
    return out


def user_message(question: str, intent: RetrievalIntent, providers: list[RankedProvider]) -> str:
    return json.dumps({
        "question": question,
        "query_understood_as": describe_intent(intent),
        "ranked_providers": prepare_records(providers),
        "required_disclaimer": DISCLAIMER,
    }, ensure_ascii=False)


# ---------- deterministic fallback ----------

def _comparison(label, p, nxt_label, nxt) -> str | None:
    if nxt is None:
        return None
    gap = score_gap(p, nxt)
    ahead = sorted(((v, d) for d, v in gap["by_dimension"].items() if v > 0), reverse=True)
    if gap["decided_by_tie_break"] and not gap["by_dimension"]:
        return f"{label} and {nxt_label} tie on every scored factor ({p.scores.total}); order set by the deterministic tie-break (name)."
    if gap["decided_by_tie_break"]:
        return f"{label} and {nxt_label} have the same total ({p.scores.total}); {label} is placed first by the tie-break order."
    lead = ", ".join(f"{DIM_LABEL[d]} (+{v:g})" for v, d in ahead[:2]) or "combined weighting"
    return f"{label} scores {p.scores.total} vs {nxt.scores.total} for {nxt_label}, ahead on {lead}."


def _proximity_text(p):
    f = _fact(p, "proximity")
    return f.explanation if f and f.score is not None else "No location supplied; proximity was not scored."


def _financial_text(p):
    f = _fact(p, "financial")
    if f is None or f.score is None:
        return "Price/affordability not part of this request and not verified in the source data."
    return f.explanation


def template_explanation(question: str, intent: RetrievalIntent, providers: list[RankedProvider],
                         empty_message: str | None = None) -> TopThreeExplanation:
    pairs = labelled(providers)
    items = []
    for i, (label, p) in enumerate(pairs):
        nxt = pairs[i + 1] if i + 1 < len(pairs) else (None, None)
        svc = _fact(p, "service")
        position = {"A": "the highest-ranked match in the current dataset",
                    "B": "the second-ranked match", "C": "the third-ranked match"}[label]
        items.append(ProviderExplanation(
            rank=label, source_id=p.source_id, heading=f"{label} — {HEADINGS[label]}",
            reason=f"{p.name} is {position} for this request (score {p.scores.total}/100).",
            proximity_explanation=_proximity_text(p),
            financial_explanation=_financial_text(p),
            service_explanation=svc.explanation if svc else "Service match not scored.",
            important_unknowns=list(p.unknown_facts),
            comparison_to_next=_comparison(label, p, nxt[0], nxt[1]),
            citation_source_ids=public_source_ids(p),
        ))
    return TopThreeExplanation(
        query_interpretation=describe_intent(intent),
        providers=items,
        ranking_disclaimer=empty_message if not providers and empty_message else DISCLAIMER,
    )


def parse_explanation(raw: str, providers: list[RankedProvider]) -> TopThreeExplanation:
    """Parse model JSON. Raises ValueError if the model reordered, replaced, added or
    dropped providers (fuller claim/citation checks are task 8)."""
    data = json.loads(raw)
    for item in data.get("providers", []):
        if item.get("comparison_to_next") == "":
            item["comparison_to_next"] = None
    exp = TopThreeExplanation.model_validate(data)
    expected = [(label, p.source_id) for label, p in labelled(providers)]
    got = [(e.rank, e.source_id) for e in exp.providers]
    if got != expected:
        raise ValueError(f"model changed the ranking: expected {expected}, got {got}")
    for e in exp.providers:  # headings are fixed by position, not model wording
        e.heading = f"{e.rank} — {HEADINGS[e.rank]}"
    return exp
