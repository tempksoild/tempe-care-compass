# rag/intent.py — Safety detection + structured intent parsing for retrieval.
#
# Deterministic (no LLM). Word-boundary matching avoids false hits such as
# "provider " being read as "er " (emergency room) in agent._fallback_parse.
#
# Order of use (see rag/retrieve.py: retrieve_for_question):
#   1. detect_emergency(text)  -> short-circuit before any retrieval
#   2. parse_intent(text)      -> RetrievalIntent that drives a fresh query

import re

from pydantic import BaseModel, Field

# Phrases that must short-circuit to "call 911 / 988" before retrieval.
EMERGENCY_PATTERNS = {
    "breathing": r"\b(can'?t|cannot|not|trouble|difficulty) breath(e|ing)\b",
    "chest pain": r"\bchest pain\b",
    "heart attack": r"\bheart attack\b",
    "stroke": r"\bstroke\b",
    "overdose": r"\boverdos(e|ed|ing)\b",
    "self-harm": r"\b(suicid\w*|kill (myself|me)|end my life|hurt myself|self[- ]harm)\b",
    "severe bleeding": r"\b(bleeding (heavily|a lot|won'?t stop)|severe bleeding)\b",
    "unconscious": r"\b(unconscious|unresponsive|passed out)\b",
    "seizure": r"\bseizur(e|ing)\b",
}

# Ordered: first matching category wins. More specific phrases come first.
CATEGORY_HINTS: dict[str, list[str]] = {
    "Other care": ["physical therapy", "physical therapist", "chiropractor", "chiropractic",
                   "athletic trainer", "speech therapy", "occupational therapy"],
    "Pharmacy": ["pharmacy", "pharmacist", "prescription", "prescriptions", "rx", "medication refill", "drugstore"],
    "Dental": ["dentist", "dental", "teeth", "tooth", "orthodontist"],
    "Behavioral health": ["therapist", "therapy", "counselor", "counseling", "mental health", "psychologist",
                          "psychiatrist", "psychiatry", "anxiety", "depression", "addiction", "substance use", "rehab"],
    "Imaging": ["mri", "x-ray", "xray", "imaging", "ct scan", "ultrasound", "radiology", "mammogram"],
    "Hospital": ["hospital", "emergency room", "er"],
    "Clinic / primary care": ["clinic", "doctor", "urgent care", "checkup", "check-up", "primary care",
                              "physician", "family medicine", "family doctor", "walk-in", "walk in", "pediatrician"],
}

# Procedure phrase -> (canonical name, [(billing_code_type, billing_code)]).
# Only codes present in the price data are listed; unknown codes stay empty.
PROCEDURES: dict[str, tuple[str, list[tuple[str, str]]]] = {
    "knee replacement": ("knee replacement", [("CPT", "27447")]),
    "knee arthroplasty": ("knee replacement", [("CPT", "27447")]),
    "colonoscopy": ("colonoscopy", [("CPT", "45378")]),
    "mri": ("mri", []),
    "ct scan": ("ct scan", []),
    "x-ray": ("x-ray", []),
    "xray": ("x-ray", []),
}

PRICE_PATTERNS = [r"\bcosts?\b", r"\bprices?\b", r"\bhow much\b", r"\bcheapest\b", r"\bcash price\b",
                  r"\bself[- ]pay\b", r"\bcompare prices?\b", r"\bfees?\b"]
AFFORDABILITY_PATTERNS = [r"\bcheap\w*\b", r"\baffordable\b", r"\blow[- ]cost\b", r"\buninsured\b",
                          r"\bno insurance\b", r"\bwithout insurance\b", r"\bsliding[- ]?(fee|scale)\b",
                          r"\bfree\b", r"\blow[- ]income\b", r"\bcharity\b", r"\bfinancial assistance\b"]
# Explicit request to restrict results to verified-affordable providers (hard filter).
VERIFIED_ONLY_PATTERNS = [
    r"\bonly\b[^.?!]{0,40}\b(verified|sliding[- ]?(fee|scale)|charity|free)\b",
    r"\b(verified|sliding[- ]?(fee|scale))\b[^.?!]{0,40}\bonly\b",
    r"\bmust (be|have|offer)\b[^.?!]{0,30}\b(sliding[- ]?(fee|scale)|charity|free)\b",
]

STOPWORDS = set("""
a an the i im i'm me my we our you your is are was be been am do does did can could would should will
need needs want looking look find get see go to for of in on at near nearby around by with without from
and or but if any some some one someone place places where what which who how please help find
tempe az arizona zip code area close closest best good near me open today now
cheap cheaper cheapest affordable low cost free uninsured insurance only verified sliding fee scale
price prices cost costs much
""".split())

ZIP_RE = re.compile(r"\b(85\d{3})\b")


class RetrievalIntent(BaseModel):
    raw_query: str
    category: str = "All"
    category_term: str | None = None          # phrase that triggered the category
    zip_code: str | None = None
    keywords: list[str] = Field(default_factory=list)
    price_query: bool = False
    procedure: str | None = None
    billing_codes: list[tuple[str, str]] = Field(default_factory=list)
    affordability_required: bool = False      # affordability is a ranking factor
    verified_only: bool = False               # affordability becomes a hard filter
    emergency: bool = False
    emergency_reason: str | None = None
    excluded_categories: list[str] = Field(default_factory=list)  # set by rag/understand.py for symptom requests
    term_priority: bool = False      # keywords are ordered best-first (from a QueryPlan)


def _has(pattern: str, text: str) -> bool:
    return re.search(pattern, text) is not None


def _phrase(term: str) -> str:
    return r"\b" + re.escape(term).replace(r"\ ", r"[\s-]+") + r"\b"


def detect_emergency(text: str) -> str | None:
    low = text.lower()
    for reason, pattern in EMERGENCY_PATTERNS.items():
        if _has(pattern, low):
            return reason
    return None


def _detect_category(low: str) -> tuple[str, str | None]:
    for category, terms in CATEGORY_HINTS.items():
        for term in terms:
            if _has(_phrase(term), low):
                return category, term
    return "All", None


def _detect_procedure(low: str):
    for phrase, (name, codes) in PROCEDURES.items():
        if _has(_phrase(phrase), low):
            return name, list(codes)
    return None, []


def _keywords(low: str) -> list[str]:
    tokens = re.findall(r"[a-z][a-z'-]*", ZIP_RE.sub(" ", low))
    seen, out = set(), []
    for t in tokens:
        t = t.strip("'-")
        if len(t) < 2 or t in STOPWORDS or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def parse_intent(text: str) -> RetrievalIntent:
    low = (text or "").lower()
    reason = detect_emergency(low)
    category, term = _detect_category(low)
    procedure, codes = _detect_procedure(low)
    zip_match = ZIP_RE.search(low)
    price_query = any(_has(p, low) for p in PRICE_PATTERNS)
    verified_only = any(_has(p, low) for p in VERIFIED_ONLY_PATTERNS)
    affordability = verified_only or any(_has(p, low) for p in AFFORDABILITY_PATTERNS)
    return RetrievalIntent(
        raw_query=text or "",
        category=category,
        category_term=term,
        zip_code=zip_match.group(1) if zip_match else None,
        keywords=_keywords(low),
        price_query=price_query,
        procedure=procedure if price_query else None,
        billing_codes=codes if price_query else [],
        affordability_required=affordability,
        verified_only=verified_only,
        emergency=reason is not None,
        emergency_reason=reason,
    )
