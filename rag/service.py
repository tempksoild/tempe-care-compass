# rag/service.py — End-to-end Top-3 recommendation used by app.py.
#
#   question -> safety -> intent -> retrieve (40) -> eligibility filter -> rank
#            -> select ≤3 (A/B/C, never padded) -> explain (LLM or template) -> validate
#
# The agent (CortexAgent / CareAgent) only explains; it is passed per call so the
# cached service (corpus + indexes) is shared across AI backends.

from datetime import date

from rag import explain as rx
from rag.config import WEIGHTS
from rag.filter import EligibilityFilter
from rag.intent import parse_intent
from rag.prices import build_price_index, hospital_price_scores
from rag.rank import rank_providers
from rag.retrieve import ProviderRetriever
from rag.schemas import RANK_LABELS, EvidenceState, TopProviderResponse
from rag.select import no_eligible_message
from rag.understand import SYMPTOM_PATTERNS, fallback_plan, merge_plan, needs_understanding
from rag.validate import validate

EMERGENCY_MESSAGE = ("Your message may describe an emergency. Call 911 now. "
                     "For a mental-health crisis, call or text 988.")
DIM_NAMES = {"service": "Service match", "proximity": "Location", "financial": "Price/affordability",
             "confidence": "Data confidence", "availability": "Verified availability"}


class ProviderRecommendationService:
    def __init__(self, repo, retriever: ProviderRetriever | None = None, require_zip: bool = False):
        self.repo = repo
        self.retriever = retriever or ProviderRetriever(repo)
        self.require_zip = require_zip
        self._prices = None

    def _price_rows(self):
        if self._prices is None:
            self._prices = self.repo.all_prices() if hasattr(self.repo, "all_prices") else []
        return self._prices

    def recommend(self, question: str, agent=None, conversation_context: list | None = None,
                  today: date | None = None, user_coords: tuple[float, float] | None = None,
                  geocoder=None) -> TopProviderResponse:
        """geocoder: optional callable(address, city, state, zip_code) -> {latitude, longitude, source,
        formatted_address} (app passes geocoding.geocode_address). Only called when the request
        contains a location the LLM/fallback extracted."""
        intent = parse_intent(question)
        if intent.emergency:  # deterministic safety first, before any LLM call
            return self._emergency(intent)

        plan, location_note = None, None
        if needs_understanding(question):
            plan = (agent.understand_request(question) if agent is not None and hasattr(agent, "understand_request")
                    else fallback_plan(question))
            intent = merge_plan(intent, plan, symptom_request=bool(SYMPTOM_PATTERNS.search(question)))
            if intent.emergency:
                return self._emergency(intent, plan)
            if user_coords is None:
                user_coords, location_note = self._locate(plan, intent, geocoder)
        if location_note is None and intent.zip_code:
            location_note = f"Searching near ZIP {intent.zip_code} (ZIP center)."
        understood = rx.describe_intent(intent)

        retrieved = self.retriever.retrieve(intent)

        hospital_prices, price_index = None, None
        if intent.price_query and intent.billing_codes:
            rows = self._price_rows()
            hospital_prices = hospital_price_scores(rows, intent.billing_codes)
            idx = build_price_index(retrieved.candidates, rows)
            price_index = idx or None  # no provider carries a CCN -> skip procedure filter

        filt = EligibilityFilter(require_zip=self.require_zip, price_index=price_index)
        eligible = filt.apply(retrieved.candidates, intent)
        ranked = rank_providers(eligible.eligible, intent, today, user_coords, hospital_prices)
        selected = [p.model_copy(update={"rank": label}) for label, p in zip(RANK_LABELS, ranked)]

        limitations = list(eligible.limitations)
        empty_message = None
        if not selected:
            empty_message = no_eligible_message(eligible.eliminated_by, eligible.considered, retrieved.note)
            limitations.insert(0, empty_message)
        elif len(selected) < 3:
            limitations.insert(0, f"Only {len(selected)} provider{'s' if len(selected) > 1 else ''} "
                                  "met the requirements; no others were added.")
        limitations += _data_limitations(selected, intent, user_coords is not None)

        if agent is not None and hasattr(agent, "explain_ranking"):
            exp = agent.explain_ranking(question, intent, selected, empty_message)
            source = getattr(agent, "last_explain_source", "template")
        else:
            exp, source = rx.template_explanation(question, intent, selected, empty_message), "template"
        if selected and validate(exp, selected):  # final guard, independent of the agent
            exp, source = rx.template_explanation(question, intent, selected, empty_message), "template"

        if plan and plan.urgency == "urgent":
            limitations.insert(0, "This may need same-day care. If symptoms are severe or getting worse, "
                                  "go to an emergency room or call 911.")

        return TopProviderResponse(
            query_understood_as=understood,
            ranking_basis=_ranking_basis(selected),
            providers=selected,
            explanations=exp.providers,
            limitations=limitations,
            eliminated_by=eligible.eliminated_by,
            explanation_source=source,
            ranking_disclaimer=rx.DISCLAIMER if selected else "",
            care_guidance=_care_guidance(plan),
            location_used=location_note,
            user_coordinates=user_coords,
            understanding_source=plan.source if plan else None,
        )

    def _emergency(self, intent, plan=None) -> TopProviderResponse:
        return TopProviderResponse(query_understood_as=rx.describe_intent(intent), ranking_basis=[], providers=[],
                                   emergency=True, limitations=[EMERGENCY_MESSAGE],
                                   understanding_source=plan.source if plan else None)

    @staticmethod
    def _locate(plan, intent, geocoder):
        """User location from the plan. A geocoder hit counts as coordinates only when it is
        a real geocode, not the geocoder's own ZIP/Tempe-centroid fallback."""
        if plan.location_text and geocoder is not None:
            try:
                g = geocoder(address=plan.location_text, city="Tempe", state="AZ", zip_code=intent.zip_code or "")
            except Exception:
                g = None
            if g and "centroid" not in str(g.get("source", "")).lower():
                return ((float(g["latitude"]), float(g["longitude"])),
                        f"Searching near {plan.location_text} (located via {g.get('source', 'geocoder')}).")
        if plan.location_text:
            where = f"ZIP {intent.zip_code}" if intent.zip_code else "Tempe"
            return None, f"Could not pinpoint '{plan.location_text}'; searching {where} instead."
        return None, None


def _care_guidance(plan) -> str | None:
    if not plan or not plan.care_type:
        return None
    text = f"Looking for: {plan.care_type}."
    if plan.reason:
        text += f" {plan.reason}"
    return text


def _ranking_basis(selected) -> list[str]:
    if not selected:
        return []
    s = selected[0].scores
    used = {"service": s.service_match, "proximity": s.proximity, "financial": s.financial,
            "confidence": s.confidence, "availability": s.availability}
    active = {k: WEIGHTS[k] for k, v in used.items() if v is not None}
    total = sum(active.values())
    return [f"{DIM_NAMES[k]} {round(100 * w / total)}%" for k, w in active.items()]


def _data_limitations(selected, intent, have_user_point: bool = False) -> list[str]:
    out = []
    if any(p.scores.distance_method == "zip_centroid" for p in selected):
        out.append("Distances are from your location to each provider's ZIP-code center (NPPES has no street coordinates)."
                   if have_user_point else "Distances are measured between ZIP-code centers, not street addresses.")
    if selected and all(p.scores.financial_evidence_state in (EvidenceState.UNKNOWN, EvidenceState.NOT_APPLICABLE)
                        for p in selected):
        out.append("No verified price or affordability program for these providers; call to ask about "
                   "self-pay and sliding-fee options.")
    if intent.price_query:
        out.append("Published hospital prices in this dataset are a California sample and are not used to rank Tempe providers.")
    if selected and all(p.scores.availability is None for p in selected):
        out.append("New-patient availability, walk-in service and hours are not verified.")
    return out
