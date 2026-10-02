# rag/understand.py — LLM query understanding for symptom / location requests.
#
# When the user describes a symptom or condition ("fever", "broke my arm", "sick")
# or gives an address/location, the LLM reads the WHOLE request and returns a
# structured QueryPlan: which kind of care to look for, NPPES specialty search
# terms, urgency, and the location text. Code then validates the plan, merges it
# into the RetrievalIntent and runs the RAG retrieval with those terms.
#
# The LLM never diagnoses and never picks providers. It only translates the request
# into search parameters. Deterministic regex safety detection runs BEFORE it, and
# an "emergency" urgency from the LLM also short-circuits to 911.
# Without an LLM (or if it fails) fallback_plan() uses a keyword table.

import json
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from rag.intent import ZIP_RE, RetrievalIntent

CATEGORIES = ["All", "Clinic / primary care", "Pharmacy", "Dental", "Behavioral health",
              "Imaging", "Hospital", "Other care"]
# NPPES files Orthopedic / Sports Medicine / Pediatrics / Emergency mostly under
# "Other care", so a symptom plan only hard-filters on categories that are clean.
STRICT_CATEGORIES = {"Dental", "Pharmacy", "Behavioral health"}
MAX_TERMS = 8

SYMPTOM_PATTERNS = re.compile(
    r"\b(fever|feverish|sick|ill|flu|cold|cough(ing)?|sore throat|strep|headache|migraine|nause\w*|vomit\w*|"
    r"diarrh\w*|rash|itch\w*|infection|infected|uti|ear ?ache|earache|pain|hurts?|hurting|ache|aching|"
    r"injur\w*|broke|broken|fractur\w*|sprain\w*|twist\w*|swollen|swelling|cut|burn(ed|t)?|bleeding|"
    r"bruise\w*|wound|stitches|allerg\w*|asthma|wheez\w*|dizz\w*|faint\w*|symptoms?|condition|"
    r"pregnan\w*|toothache|tooth ache|lump|bump|bite|sting|concussion|back pain|stomach|"
    r"arm|arms|leg|legs|ankle|wrist|knee|elbow|shoulder|hip|foot|feet|hand|finger|toe|bones?|skin|eyes?|"
    r"ears?|throat|neck|head|chest|belly|abdomen)\b", re.I)
LOCATION_PATTERNS = re.compile(
    r"\b(address|location|located|i live|live (at|on|near)|i'?m (at|on|near)|staying at|near( me)?|around|"
    r"close to|nearby|walking distance|by (the|my)|street|st\.?|ave(nue)?|blvd|boulevard|rd\.?|road|"
    r"dr\.?|drive|lane|ln|way|campus|asu|mill ave|downtown|apartment|apt|dorm|hotel)\b", re.I)


def needs_understanding(text: str) -> bool:
    """True when the request mentions a symptom/condition or a location/address."""
    return bool(SYMPTOM_PATTERNS.search(text or "") or LOCATION_PATTERNS.search(text or ""))


class QueryPlan(BaseModel):
    care_type: str = ""                         # plain-language, e.g. "urgent care or orthopedics"
    category: str = "All"
    search_terms: list[str] = Field(default_factory=list)
    urgency: Literal["emergency", "urgent", "routine", "unknown"] = "unknown"
    location_text: str = ""                     # address / landmark as written by the user
    zip_code: str = ""
    reason: str = ""                            # why this care type (navigation, not diagnosis)
    source: str = "fallback"                    # "llm" | "fallback"

    @field_validator("category")
    @classmethod
    def _cat(cls, v):
        return v if v in CATEGORIES else "All"

    @field_validator("search_terms")
    @classmethod
    def _terms(cls, v):
        out = []
        for t in v:
            t = re.sub(r"[^a-zA-Z /&'-]", "", str(t)).strip().lower()
            if 2 < len(t) <= 40 and t not in out:
                out.append(t)
        return out[:MAX_TERMS]

    @field_validator("zip_code")
    @classmethod
    def _zip(cls, v):
        m = ZIP_RE.search(str(v or ""))
        return m.group(1) if m else ""

    @field_validator("location_text")
    @classmethod
    def _loc(cls, v):
        return str(v or "").strip()[:120]


UNDERSTAND_PROMPT = """You turn a healthcare-navigation request into search parameters for a Tempe, AZ provider directory (CMS NPPES data).
Read the whole request. Do not diagnose, do not give medical advice, do not name providers.

Return:
- care_type: plain-language kind of care to look for (e.g. "urgent care", "urgent care or orthopedics", "primary care", "dentist").
- category: one of All, Clinic / primary care, Pharmacy, Dental, Behavioral health, Imaging, Hospital, Other care.
  Use All unless the need is clearly dental, pharmacy or behavioral health.
- search_terms: up to 8 lowercase NPPES specialty/taxonomy words to search, e.g. "urgent care", "family", "primary care",
  "internal medicine", "orthopedic", "sports medicine", "pediatrics", "dermatology", "emergency", "general practice".
- urgency: "emergency" for possibly life-threatening situations (trouble breathing, chest pain, severe bleeding,
  stroke signs, unconsciousness, suicidal intent, a bone visibly out of place); "urgent" for same-day needs
  (fever, possible fracture, sprain, infection, cut needing stitches); "routine" otherwise; "unknown" if unclear.
- location_text: the address, street, intersection or landmark the user gave, copied as written; "" if none.
- zip_code: 5-digit ZIP if the user gave one, else "".
- reason: one short sentence on why that care type fits the request, phrased as navigation, not diagnosis.
Return JSON matching the schema exactly."""

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "care_type": {"type": "string"},
        "category": {"type": "string", "enum": CATEGORIES},
        "search_terms": {"type": "array", "items": {"type": "string"}},
        "urgency": {"type": "string", "enum": ["emergency", "urgent", "routine", "unknown"]},
        "location_text": {"type": "string"},
        "zip_code": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["care_type", "category", "search_terms", "urgency", "location_text", "zip_code", "reason"],
    "additionalProperties": False,
}

# Keyword fallback: pattern -> (care_type, category, terms, urgency)
_FALLBACK = [
    (r"\b(broke|broken|fractur\w*|sprain\w*|twist\w*|dislocat\w*|injur\w*|swollen (ankle|wrist|knee))\b",
     "urgent care or orthopedics", "All", ["urgent care", "orthopedic", "sports medicine", "emergency"], "urgent"),
    (r"\b(cut|stitches|wound|burn(ed|t)?|bite|sting)\b",
     "urgent care", "All", ["urgent care", "emergency", "family"], "urgent"),
    (r"\b(fever|feverish|flu|cold|cough(ing)?|sore throat|strep|sick|ill|infection|infected|uti|ear ?ache|earache|"
     r"nause\w*|vomit\w*|diarrh\w*|stomach)\b",
     "urgent care or primary care", "All", ["urgent care", "family", "primary care", "internal medicine",
                                           "general practice"], "urgent"),
    (r"\b(rash|itch\w*|skin)\b", "primary care or dermatology", "All",
     ["dermatology", "family", "primary care", "urgent care"], "routine"),
    (r"\b(allerg\w*|asthma|wheez\w*)\b", "primary care or allergy", "All",
     ["allergy", "family", "primary care", "urgent care"], "routine"),
    (r"\b(toothache|tooth ache|tooth|teeth|gum)\b", "dentist", "Dental", ["general practice", "dental"], "routine"),
    (r"\b(pregnan\w*)\b", "obstetrics", "All", ["obstetrics", "gynecology", "family"], "routine"),
    (r"\b(kid|child|baby|toddler|son|daughter)\b", "pediatrics", "All", ["pediatrics", "family"], "unknown"),
    (r"\b(back pain|joint pain|knee pain|pain|ache|aching|hurts?)\b", "primary care", "All",
     ["family", "primary care", "sports medicine", "physical therapy"], "routine"),
]
_LOCATION_TEXT = re.compile(
    r"\b(\d{1,6}\s+[a-z0-9 .'-]{2,40}?\b(st|street|ave|avenue|blvd|boulevard|rd|road|dr|drive|ln|lane|way|pkwy|parkway)\b\.?)",
    re.I)
_LANDMARK = re.compile(r"\b(asu|arizona state university|mill ave(nue)?|tempe town lake|downtown tempe)\b", re.I)


def fallback_plan(text: str) -> QueryPlan:
    low = (text or "").lower()
    plan = QueryPlan(source="fallback")
    terms: list[str] = []
    for pattern, care, cat, t, urg in _FALLBACK:
        if re.search(pattern, low):
            if not plan.care_type:
                plan.care_type, plan.category, plan.urgency = care, cat, urg
                plan.reason = f"Requests like this are usually handled by {care}."
            terms += t
    plan.search_terms = QueryPlan._terms(terms)
    loc = _LOCATION_TEXT.search(text or "") or _LANDMARK.search(text or "")
    plan.location_text = loc.group(0).strip() if loc else ""
    z = ZIP_RE.search(text or "")
    plan.zip_code = z.group(1) if z else ""
    return plan


def parse_plan(raw: str) -> QueryPlan:
    data = json.loads(raw)
    return QueryPlan(**{k: data.get(k, "") if k != "search_terms" else data.get(k, []) for k in PLAN_SCHEMA["properties"]},
                     source="llm")


def merge_plan(intent: RetrievalIntent, plan: QueryPlan, symptom_request: bool) -> RetrievalIntent:
    """Fold the plan into the deterministic intent. Plan terms lead retrieval; the
    user's own keywords are kept so nothing they said is dropped."""
    category = plan.category
    if symptom_request and category not in STRICT_CATEGORIES:
        category = "All"
    if category == "All" and intent.category in STRICT_CATEGORIES:
        category = intent.category  # user explicitly asked for a dentist/pharmacy/therapist
    # Symptom words ("broke", "fever", "daughter") never appear in NPPES specialties; when the
    # plan supplies specialty terms, search with those only. Otherwise keep the user's words too.
    if symptom_request and plan.search_terms:
        keywords = list(plan.search_terms)
    else:
        keywords = list(dict.fromkeys(plan.search_terms + intent.keywords))
    # Physical symptoms searched across "All": keep dental-orthodontics, pharmacies and
    # "family therapists" out of a fever / broken-arm result.
    excluded = sorted(STRICT_CATEGORIES) if symptom_request and category == "All" else []
    return intent.model_copy(update={
        "category": category,
        "excluded_categories": excluded,
        "term_priority": bool(plan.search_terms),
        "category_term": intent.category_term if category == intent.category else None,
        "keywords": keywords,
        "zip_code": intent.zip_code or plan.zip_code or None,
        "emergency": intent.emergency or plan.urgency == "emergency",
        "emergency_reason": intent.emergency_reason or ("assessed as emergency" if plan.urgency == "emergency" else None),
    })
