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
                  today: date | None = None, user_coords: tuple[float, float] | None = None) -> TopProviderResponse:
        intent = parse_intent(question)
        understood = rx.describe_intent(intent)
        if intent.emergency:
            return TopProviderResponse(query_understood_as=understood, ranking_basis=[], providers=[],
                                       emergency=True, limitations=[EMERGENCY_MESSAGE])

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
        limitations += _data_limitations(selected, intent)

        if agent is not None and hasattr(agent, "explain_ranking"):
            exp = agent.explain_ranking(question, intent, selected, empty_message)
            source = getattr(agent, "last_explain_source", "template")
        else:
            exp, source = rx.template_explanation(question, intent, selected, empty_message), "template"
        if selected and validate(exp, selected):  # final guard, independent of the agent
            exp, source = rx.template_explanation(question, intent, selected, empty_message), "template"

        return TopProviderResponse(
            query_understood_as=understood,
            ranking_basis=_ranking_basis(selected),
            providers=selected,
            explanations=exp.providers,
            limitations=limitations,
            eliminated_by=eligible.eliminated_by,
            explanation_source=source,
            ranking_disclaimer=rx.DISCLAIMER if selected else "",
        )


def _ranking_basis(selected) -> list[str]:
    if not selected:
        return []
    s = selected[0].scores
    used = {"service": s.service_match, "proximity": s.proximity, "financial": s.financial,
            "confidence": s.confidence, "availability": s.availability}
    active = {k: WEIGHTS[k] for k, v in used.items() if v is not None}
    total = sum(active.values())
    return [f"{DIM_NAMES[k]} {round(100 * w / total)}%" for k, w in active.items()]


def _data_limitations(selected, intent) -> list[str]:
    out = []
    if any(p.scores.distance_method == "zip_centroid" for p in selected):
        out.append("Distances are measured between ZIP-code centers, not street addresses.")
    if selected and all(p.scores.financial_evidence_state in (EvidenceState.UNKNOWN, EvidenceState.NOT_APPLICABLE)
                        for p in selected):
        out.append("No verified price or affordability program for these providers; call to ask about "
                   "self-pay and sliding-fee options.")
    if intent.price_query:
        out.append("Published hospital prices in this dataset are a California sample and are not used to rank Tempe providers.")
    if selected and all(p.scores.availability is None for p in selected):
        out.append("New-patient availability, walk-in service and hours are not verified.")
    return out
