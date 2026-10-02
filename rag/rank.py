# rag/rank.py — Deterministic scoring + tie-breaking. The LLM never decides order.
#
# total = weighted mean of available dimensions (0-100 each), weights in rag/config.py.
# Missing dimensions (None) are dropped and the remaining weights renormalized,
# EXCEPT financial: when the request cares about cost and evidence is unknown it
# gets NEUTRAL_FINANCIAL (30) so unknown affordability is not treated as irrelevant.
#
# Output is the full ordered list of RankedProvider (rank=None). A/B/C labels are
# assigned in task 6. Proximity: rag/geo.py (task 9). Prices: rag/prices.py (task 10).

import re
from datetime import date

from rag import config as cfg
from rag.citations import normalize_ccn, user_location_id
from rag.evidence import affordability_evidence, availability_evidence, is_active
from rag.geo import haversine_miles, record_coordinates, user_location, zip_centroid  # noqa: F401 (re-export)
from rag.intent import RetrievalIntent
from rag.prices import HospitalPrice
from rag.retrieve import SERVICE_SYNONYMS, Candidate
from rag.schemas import EvidenceState, RankedProvider, RankingBreakdown, ScoredFact

FINANCIAL_PRIORITY = {
    EvidenceState.VERIFIED_YES: 2,
    EvidenceState.UNKNOWN: 1,
    EvidenceState.NOT_APPLICABLE: 1,
    EvidenceState.VERIFIED_NO: 0,
}


def weighted_score(scores: dict[str, float | None], weights: dict[str, float] = cfg.WEIGHTS) -> float:
    available = {k: v for k, v in scores.items() if v is not None}
    weight_sum = sum(weights[k] for k in available)
    if weight_sum == 0:
        return 0.0
    return sum(weights[k] * v for k, v in available.items()) / weight_sum


def proximity_score(distance_miles: float | None) -> float:
    if distance_miles is None:
        return 40
    for limit, score in ((1, 100), (3, 85), (5, 70), (10, 45), (20, 20)):
        if distance_miles <= limit:
            return score
    return 0



def _contains(text: str, term: str) -> bool:
    return re.search(r"\b" + re.escape(term.lower()).replace(r"\ ", r"[\s/-]+"), text.lower()) is not None


def request_terms(intent: RetrievalIntent) -> list[str]:
    terms = list(intent.keywords)
    for kw in intent.keywords:
        terms += SERVICE_SYNONYMS.get(kw, [])
    if intent.category_term:
        terms += [intent.category_term] + SERVICE_SYNONYMS.get(intent.category_term, [])
    return [t for t in dict.fromkeys(terms) if len(t) >= 3]


def term_positions(intent: RetrievalIntent) -> dict[str, int]:
    """term (incl. synonyms) -> position of the keyword it came from."""
    pos: dict[str, int] = {}
    for i, kw in enumerate(intent.keywords):
        for t in [kw] + SERVICE_SYNONYMS.get(kw, []):
            pos.setdefault(t, i)
    return pos


def service_fact(c: Candidate, intent: RetrievalIntent) -> ScoredFact:
    r, sid = c.record, c.source_id
    specialty, name = str(r.get("specialty") or ""), str(r.get("name") or "")
    terms = request_terms(intent)
    cat_ok = intent.category == "All" or r.get("category") == intent.category
    spec_hit = next((t for t in terms if specialty and _contains(specialty, t)), None)
    name_hit = next((t for t in terms if _contains(name, t)), None)

    hit_positions = ([p for t, p in term_positions(intent).items() if _contains(specialty, t)]
                     if cat_ok and spec_hit and intent.term_priority else [])
    if hit_positions:
        # Plan terms are ordered best-first by the understanding step ("urgent care" before
        # "orthopedic"): the best-positioned matching term (synonyms inherit their source
        # term's position) sets the score, floored at the category+keyword tier.
        score = max(cfg.SERVICE_CATEGORY_KEYWORD,
                    cfg.SERVICE_SPECIALTY_MATCH - cfg.SERVICE_TERM_STEP * min(hit_positions))
        why = f"Specialty '{specialty}' matches requested '{spec_hit}'"
    elif cat_ok and spec_hit:
        score, why = cfg.SERVICE_SPECIALTY_MATCH, f"Specialty '{specialty}' matches requested '{spec_hit}'"
    elif cat_ok and name_hit and intent.category != "All":
        score, why = cfg.SERVICE_CATEGORY_KEYWORD, f"{r.get('category')} category and name matches '{name_hit}'"
    elif cat_ok and intent.category != "All":
        score, why = cfg.SERVICE_CATEGORY_ONLY, f"{r.get('category')} category matches the request"
    elif c.semantic_score:
        frac = min(1.0, c.semantic_score / cfg.SERVICE_SEMANTIC_FULL_AT)
        score = round(cfg.SERVICE_SEMANTIC_MIN + frac * (cfg.SERVICE_SEMANTIC_MAX - cfg.SERVICE_SEMANTIC_MIN), 1)
        why = "Related service by text similarity only"
    elif name_hit or c.lexical_score:
        score, why = cfg.SERVICE_WEAK, "General keyword match only"
    else:
        score, why = cfg.SERVICE_NONE, "No service match found"
    state = EvidenceState.VERIFIED_YES if score >= cfg.SERVICE_CATEGORY_ONLY else EvidenceState.UNKNOWN
    return ScoredFact(name="service", score=score, explanation=why, source_ids=[sid], evidence_state=state)


def _prox(score, why, ids, state, dist, method):
    return ScoredFact(name="proximity", score=score, explanation=why, source_ids=ids, evidence_state=state), dist, method


def proximity_fact(c: Candidate, intent: RetrievalIntent, user_coords: tuple[float, float] | None):
    """Returns (fact, distance_miles, distance_method). Fallback chain:
      1. provider_coordinates  exact provider + user coordinates        -> miles
      2. zip_centroid          ZIP-center to ZIP-center (different ZIPs) -> approx miles
      3. same_zip              same ZIP as requested                     -> no miles
      4. same_city             in Tempe                                  -> no miles
      5. unknown
    Miles are never derived from a ZIP *match*; same-ZIP centroids (0 mi) use same_zip."""
    r, sid = c.record, c.source_id
    u_coords, u_method = user_location(intent.zip_code, user_coords)
    if u_method is None:
        return _prox(None, "No user location supplied", [], EvidenceState.NOT_APPLICABLE, None, None)
    u_id = user_location_id(u_method)
    pzip = str(r.get("zip") or "")[:5]
    exact = record_coordinates(r)

    if u_method == "user_coordinates" and exact:
        d = round(haversine_miles(u_coords, exact), 1)
        return _prox(proximity_score(d), f"Calculated distance is {d} miles", [sid, u_id],
                     EvidenceState.VERIFIED_YES, d, "provider_coordinates")
    if intent.zip_code and pzip == intent.zip_code and u_method != "user_coordinates":
        return _prox(cfg.LOCATION_MATCH_SCORES["same_zip"],
                     f"Same ZIP code as requested ({pzip}); street-level distance not measured",
                     [sid, u_id], EvidenceState.VERIFIED_YES, None, "same_zip")
    p_coords = exact or zip_centroid(pzip)
    if u_coords and p_coords:
        d = round(haversine_miles(u_coords, p_coords), 1)
        if u_method == "zip_centroid":
            why = f"About {d} miles between ZIP centers ({intent.zip_code} to {pzip}); approximate, not street-level"
        else:
            why = f"About {d} miles from your location to the center of ZIP {pzip}; approximate, not street-level"
        return _prox(proximity_score(d), why,
                     [sid, u_id, f"GEO:zcta_centroid:{pzip}"], EvidenceState.VERIFIED_YES, d, "zip_centroid")
    if str(r.get("city") or "").strip().lower() == cfg.SERVICE_AREA["city"]:
        return _prox(cfg.LOCATION_MATCH_SCORES["same_city"],
                     f"In Tempe (ZIP {pzip or 'unknown'}); distance not measured", [sid, u_id],
                     EvidenceState.VERIFIED_YES, None, "same_city")
    return _prox(cfg.LOCATION_MATCH_SCORES["unknown"], "Location relative to user unknown", [],
                 EvidenceState.UNKNOWN, None, "unknown")


def financial_fact(c: Candidate, intent: RetrievalIntent,
                   hospital_prices: dict[str, HospitalPrice] | None = None):
    """Returns (fact, extras). price_score and affordability_score are kept separate;
    financial = price_score for price questions (when a comparable local price exists),
    otherwise affordability_score. Unknown -> NEUTRAL_FINANCIAL when cost matters, else None."""
    aff = affordability_evidence(c.record)
    aff_score = aff.score if aff.state != EvidenceState.UNKNOWN else None
    important = intent.affordability_required or intent.price_query

    hp = (hospital_prices or {}).get(normalize_ccn(c.record.get("ccn"))) if c.record.get("ccn") else None
    price_score = hp.price_score if hp else None
    price_eligible = bool(hp and hp.price_eligible)

    if intent.price_query and price_score is not None:
        g = hp.group
        why = (f"Published {g[3]} price ${hp.price:,.2f} for {g[0]} {g[1]} ({g[2]}); range "
               f"${hp.group_min:,.2f}-${hp.group_max:,.2f} across {hp.hospitals_compared} hospitals")
        score, state, ids, basis = price_score, EvidenceState.VERIFIED_YES, hp.source_ids, "price"
    elif aff_score is not None:
        score, why, state, ids, basis = aff_score, aff.description, aff.state, aff.source_ids, "affordability"
    elif important:
        why = ("No comparable local published price" if intent.price_query and not intent.affordability_required
               else aff.description)
        if hp and not hp.price_eligible:
            why += f"; {hp.note.lower()}"
        why += " (unknown, scored neutral; not assumed expensive)"
        score, state, ids, basis = cfg.NEUTRAL_FINANCIAL, EvidenceState.UNKNOWN, [], "neutral_unknown"
    else:
        score, why, state, ids, basis = None, aff.description, EvidenceState.NOT_APPLICABLE, [], None

    fact = ScoredFact(name="financial", score=score, explanation=why, source_ids=ids, evidence_state=state)
    return fact, {"price_score": price_score, "affordability_score": aff_score,
                  "price_eligible": price_eligible, "financial_basis": basis}


def confidence_fact(c: Candidate, today: date) -> ScoredFact:
    r, pts, got, missing = c.record, cfg.CONFIDENCE_POINTS, [], []
    checks = {
        "active": is_active(r) == EvidenceState.VERIFIED_YES,
        "full_address": bool(str(r.get("address") or "").strip() and str(r.get("zip") or "").strip()),
        "phone": len(re.sub(r"\D", "", str(r.get("phone") or ""))) == 10,
        "specialty": bool(str(r.get("specialty") or "").strip()),
        "fresh": _fresh(r.get("last_updated"), today),
    }
    for k, ok in checks.items():
        (got if ok else missing).append(k)
    score = float(sum(pts[k] for k in got))
    why = "Record has: " + (", ".join(got) or "none") + ("; missing: " + ", ".join(missing) if missing else "")
    return ScoredFact(name="confidence", score=score, explanation=why, source_ids=[c.source_id],
                      evidence_state=EvidenceState.VERIFIED_YES)


def _fresh(value, today: date) -> bool:
    try:
        d = date.fromisoformat(str(value)[:10])
    except ValueError:
        return False
    return (today - d).days <= cfg.FRESH_YEARS * 365


def availability_fact(c: Candidate) -> ScoredFact:
    state, why = availability_evidence(c.record)
    score = {EvidenceState.VERIFIED_YES: 100.0, EvidenceState.VERIFIED_NO: 0.0}.get(state)
    return ScoredFact(name="availability", score=score, explanation=why,
                      source_ids=[c.source_id] if score is not None else [], evidence_state=state)


def score_candidate(c: Candidate, intent: RetrievalIntent, today: date,
                    user_coords: tuple[float, float] | None = None,
                    hospital_prices: dict[str, HospitalPrice] | None = None) -> RankedProvider:
    r = c.record
    service = service_fact(c, intent)
    proximity, dist, method = proximity_fact(c, intent, user_coords)
    financial, fin = financial_fact(c, intent, hospital_prices)
    confidence = confidence_fact(c, today)
    availability = availability_fact(c)
    facts = [service, proximity, financial, confidence, availability]

    total = round(weighted_score({
        "service": service.score, "proximity": proximity.score, "financial": financial.score,
        "confidence": confidence.score, "availability": availability.score,
    }), 2)

    evidence = [f.explanation for f in facts if f.evidence_state == EvidenceState.VERIFIED_YES and f.score is not None]
    unknown = [f.explanation for f in facts if f.evidence_state == EvidenceState.UNKNOWN]
    addr = ", ".join(x for x in (r.get("address"), r.get("city"), r.get("state"), r.get("zip")) if x)

    return RankedProvider(
        source_id=c.source_id, npi=str(r.get("npi") or ""), name=str(r.get("name") or ""),
        category=str(r.get("category") or ""), specialty=str(r.get("specialty") or ""),
        address=addr, phone=str(r.get("phone") or "") or None,
        scores=RankingBreakdown(
            service_match=service.score, proximity=proximity.score, financial=financial.score,
            confidence=confidence.score, availability=availability.score, total=total,
            distance_miles=dist, distance_method=method,
            financial_evidence_state=financial.evidence_state,
            availability_evidence_state=availability.evidence_state,
            **fin,
        ),
        facts=facts, evidence_facts=evidence, unknown_facts=unknown,
    )


def sort_key(p: RankedProvider):
    s = p.scores
    return (
        -s.total,
        -s.service_match,
        # "Shortest verified distance": compare the proximity score first so a same-ZIP
        # match (no miles, score 90) is not treated as infinitely far, then miles.
        -(s.proximity if s.proximity is not None else -1),
        s.distance_miles if s.distance_miles is not None else float("inf"),
        -FINANCIAL_PRIORITY.get(s.financial_evidence_state, 1),
        -s.confidence,
        p.name.lower(),
        p.source_id,  # final guard so equal names still order deterministically
    )


def rank_providers(candidates: list[Candidate], intent: RetrievalIntent, today: date | None = None,
                   user_coords: tuple[float, float] | None = None,
                   hospital_prices: dict[str, HospitalPrice] | None = None) -> list[RankedProvider]:
    """hospital_prices: {ccn: HospitalPrice} from rag.prices.hospital_price_scores(); only
    used for price questions and only for candidates whose record carries a CCN."""
    today = today or date.today()
    scored = [score_candidate(c, intent, today, user_coords, hospital_prices) for c in candidates]
    return sorted(scored, key=sort_key)
