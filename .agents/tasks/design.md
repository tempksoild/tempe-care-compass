# Design: Deterministic Top-3 Provider Ranking (backend only)

Source of truth: `.agents/tasks/top3-ranking-spec.md`. Where this document and the spec disagree, the spec wins; every intentional deviation is listed in §13.

Central rule: retrieval finds candidates, deterministic code ranks them, the LLM only explains the order. No UI/visual change; `app.py` swaps its AI-tab orchestration for `result = recommendation_service.recommend(q)`.

## 1. Facts about the current data that drive the design

Checked against `data/*.csv` with the project `.venv` (Python 3.11.15, pandas, pydantic 2 present; pytest not installed):

| Fact | Consequence |
|---|---|
| 6,183 providers, all `city=Tempe`, `state=AZ`, NPI unique, no duplicate NPI+address | Dedupe still required (Snowflake rows can repeat per taxonomy/address), but demo will rarely trigger it |
| 4,952 rows (80%) have empty `address`; 3,827 (62%) empty `specialty`; 2 empty `phone` | Missing address/specialty/phone must lower **confidence**, not be a hard filter, or most of the corpus disappears |
| No latitude/longitude columns anywhere | Proximity distance is `None` for all rows today → score 40 when location matters; never claim miles |
| `affordability` is always `"Not stated in NPPES — call to verify"`; `VERIFIED_ACCESS_PROGRAMS` exists only in Snowflake and is empty in the demo | Affordability evidence state = `unknown` for everyone unless a verified-program row exists |
| Prices: 33 CCNs, all prefixed `05` (California), single `snapshot_date` 2026-05-22, CPT 27447 and 45378 only | Price data is the California sample → `price_eligible=False`, `price_score=None` for Tempe ranking |
| Curated table filters `IS_ACTIVE = TRUE`; no status column is exported | Active status is inferred from the curated source (see A1) |

## 2. Module layout

```
rag/
  __init__.py
  config.py          # weights, thresholds, synonym map, tunables (single source)
  schemas.py         # EvidenceState, all pydantic models
  ingest.py          # raw rows -> ProviderRecord / PriceRecord / AccessProgram, source IDs, normalization
  retrieve.py        # corpus scoping, BM25 lexical, local semantic, RRF, dedupe
  rank.py            # EligibilityFilter, component scorers, weighted_score, tie-break, Ranker
  citations.py       # source-ID builders/parsers, citation allow-list, ScoredFact builders
  validate.py        # explanation validator + claim checker
  service.py         # IntentParser adapter, ExplanationGenerator, ProviderRecommendationService, factory
llm/
  __init__.py
  cortex_messages.py # Cortex Messages API client (REST, structured output via output_config)
tests/
  conftest.py        # fixture builders for ProviderRecord/PriceRecord, fake LLM clients
  test_ranking.py
  test_missing_data.py
  test_top_three.py
  test_citations.py
  test_price_matching.py
requirements-dev.txt # pytest pinned (exact version)
```

Decision: config lives in `rag/config.py` as module-level frozen dataclasses (no new YAML dependency). Weights may be overridden by env var `RANKING_WEIGHTS_JSON` for tuning runs; invalid JSON or weights not summing to 1.0 ± 1e-6 raise at import time.

Existing `agent.py` and `repository.py` are kept. `agent._fallback_parse` / `CareAgent.parse` remain the intent source (extended, not replaced). `repository.py` gains one read method per repository class (§5.1). Nothing is deleted.

## 3. Configuration (`rag/config.py`)

```python
@dataclass(frozen=True)
class RankingWeights:
    service_match: float = 0.35
    proximity: float = 0.30
    financial: float = 0.20
    confidence: float = 0.10
    availability: float = 0.05

@dataclass(frozen=True)
class RetrievalConfig:
    lexical_top_k: int = 50
    semantic_top_k: int = 50
    rrf_k: int = 60            # standard RRF constant
    pool_size: int = 40
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

@dataclass(frozen=True)
class ScoringConfig:
    unknown_distance_score: float = 40.0
    unknown_financial_score: float = 30.0      # neutral-low, never 0 for unknown
    freshness_years: int = 2
    max_selected: int = 3

WEIGHTS: RankingWeights
RETRIEVAL: RetrievalConfig
SCORING: ScoringConfig
SERVICE_SYNONYMS: dict[str, list[str]]      # "therapist" -> ["therapy","counsel","mental health","psycholog","behavioral"]
SPECIALTY_TERMS: dict[str, list[str]]       # query term -> exact specialty values, e.g. "psychiatrist" -> ["Psychiatry","Psychiatric/Mental Health"]
PROCEDURE_CODES: dict[str, tuple[str, str]] # "colonoscopy" -> ("45378","CPT"), "knee replacement" -> ("27447","CPT")
LOCAL_PRICE_STATES: frozenset[str] = {"AZ"} # price rows outside these states are not price-eligible
CCN_STATE_PREFIX: dict[str, str] = {"03": "AZ", "05": "CA"}   # CMS CCN first two digits = state code
```

## 4. Data models (`rag/schemas.py`)

All models are pydantic v2 (`BaseModel`, `model_config = ConfigDict(extra="forbid")` on LLM-facing models).

```python
class EvidenceState(str, Enum):
    VERIFIED_YES = "verified_yes"
    VERIFIED_NO = "verified_no"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"

class AffordabilityEvidence(str, Enum):      # ordered by tie-break priority, highest first
    FREE_PROGRAM = "verified_free_program"          # 100
    SLIDING_FEE = "verified_sliding_fee"            # 90
    DISCOUNTED_SELF_PAY = "verified_discounted_self_pay"  # 80
    CASH_PRICE_ONLY = "verified_cash_price_no_assistance" # 60
    UNKNOWN = "unknown"                             # 30
    NO_ASSISTANCE = "verified_no_assistance"        # 0

class CareIntentV2(BaseModel):               # superset of agent.CareIntent
    raw_question: str
    category: str = "All"
    zip_code: str | None = None
    city: str | None = None
    location_required: bool = False           # True when the user named a ZIP/city/"near me"+ZIP
    zip_strict: bool = False                  # "only in 85281" -> other ZIPs are hard-filtered (A3)
    emergency: bool = False
    keywords: list[str] = []                  # service terms after stop-word removal
    price_query: bool = False
    procedure: str | None = None
    billing_code: str | None = None           # resolved via PROCEDURE_CODES
    billing_code_type: str | None = None
    affordability_important: bool = False     # uninsured / low-cost / cheap / sliding / free / afford
    require_verified_affordability: bool = False  # "only verified sliding-fee", "only free clinics"
    required_affordability: AffordabilityEvidence | None = None

class ProviderRecord(BaseModel):              # normalized output of ingest
    source_id: str                            # NPPES:<npi>:address:<hash>
    npi: str
    name: str
    category: str
    specialty: str                            # "" when missing
    address: str                              # "" when missing
    city: str; state: str; zip: str
    phone: str | None
    last_updated: date | None
    active_state: EvidenceState
    latitude: float | None = None             # only from a real geocode column
    longitude: float | None = None
    coord_method: Literal["provider_coordinates"] | None = None
    affordability: AffordabilityEvidence = AffordabilityEvidence.UNKNOWN
    affordability_source_id: str | None = None    # VAP:<npi>:<hash>
    availability_state: EvidenceState = EvidenceState.UNKNOWN
    availability_source_id: str | None = None
    raw: dict                                 # original row, used by app display

class PriceRecord(BaseModel):
    source_id: str                            # HPT:<ccn>:<code_type>:<code>:<rate_type>:<hash>
    ccn: str; billing_code: str; billing_code_type: str; description: str
    payer_name: str | None; rate_type: str; setting: str | None
    rate_amount: float; snapshot_date: date | None
    state: str | None                         # derived from CCN prefix
    npi: str | None = None                    # hospital identity link if known (none today)
    price_eligible: bool

class ScoredFact(BaseModel):                  # spec model, unchanged
    name: str; score: float; explanation: str
    source_ids: list[str]; evidence_state: EvidenceState

class RankingBreakdown(BaseModel):            # spec model, unchanged
    service_match: float
    proximity: float | None
    financial: float | None
    confidence: float
    availability: float | None
    total: float
    distance_miles: float | None
    distance_method: str | None               # provider_coordinates | zip_centroid | same_zip | same_city | unknown | None(no user location)
    financial_evidence_state: str             # AffordabilityEvidence value or price state
    availability_evidence_state: str          # EvidenceState value

class RankedProvider(BaseModel):              # spec model + `facts` and `raw`
    rank: Literal["A", "B", "C"]
    source_id: str; npi: str; name: str; category: str; specialty: str
    address: str; phone: str | None
    scores: RankingBreakdown
    evidence_facts: list[str]                 # human-readable verified facts
    unknown_facts: list[str]                  # "Price unknown", "Availability not verified", ...
    facts: list[ScoredFact]                   # structured, used for LLM payload and template fallback
    raw: dict = Field(exclude=True)           # original row; not sent to the LLM

class ProviderExplanation(BaseModel):         # spec model, unchanged
    rank: Literal["A","B","C"]; source_id: str; heading: str; reason: str
    proximity_explanation: str; financial_explanation: str; service_explanation: str
    important_unknowns: list[str]; comparison_to_next: str | None
    citation_source_ids: list[str]

class TopThreeExplanation(BaseModel):         # spec model, unchanged
    query_interpretation: str
    providers: list[ProviderExplanation]
    ranking_disclaimer: str

class EliminationReport(BaseModel):
    scoped_out: dict[str, int]                # constraint name -> count removed before pooling
    filtered_out: dict[str, int]              # hard-filter name -> count removed from the pool
    eligible_count: int

class TopProviderResponse(BaseModel):         # spec fields + additive fields
    query_understood_as: str
    ranking_basis: list[str]                  # e.g. ["service match 35%", "proximity 30%", ...] after renormalization
    providers: list[RankedProvider]           # 0..3, in A/B/C order
    limitations: list[str]
    # additive (see §13 D1)
    emergency: bool = False
    explanation: TopThreeExplanation | None = None
    explanation_source: Literal["llm", "template", "none"] = "none"
    elimination: EliminationReport | None = None
    def to_markdown(self) -> str: ...         # renders the "Top 3 providers / A. ... Why: ..." block
```

Unknown-sensitive fields (`active_state`, `availability_state`, `affordability`) never use `bool`.

## 5. Pipeline

```
recommend(question)
 ├─ 1 safety:   detect_emergency(question)            -> emergency response, stop
 ├─ 2 intent:   IntentParser.parse(question)          -> CareIntentV2
 ├─ 3 retrieve: Retriever.retrieve_providers(intent)  -> list[ProviderRecord] (pool ≤ 40) + scope counts
 ├─ 4 filter:   EligibilityFilter.apply(pool, intent) -> eligible + filtered_out counts
 ├─ 5 rank:     Ranker.rank(eligible, intent)         -> list[RankedProvider] (all, sorted)
 ├─ 6 select:   ranked[:3], ranks assigned "A","B","C" by code
 ├─ 7 explain:  ExplanationGenerator.explain_ranking(question, intent, selected)
 ├─ 8 validate: ExplanationValidator.validate(explanation, selected) -> ok | fallback template
 └─ 9 return:   TopProviderResponse
```

### 5.1 Safety and intent

- `detect_emergency(text)`: reuses `_fallback_parse` phrase list (chest pain, can't breathe, overdose, suicid) plus "stroke", "unconscious", "severe bleeding". Runs before any LLM call. On hit: `TopProviderResponse(emergency=True, providers=[], limitations=["Call 911 now ..."], explanation_source="none")`. app.py keeps its existing `st.error("... Call 911 now.")` banner, driven by `result.emergency`.
- `IntentParser.parse(question) -> CareIntentV2`: calls the existing backend `parse()` (`CareAgent` LLM-structured or `_fallback_parse`) for the base `CareIntent`, then deterministically enriches: keywords (tokenize, drop stop words and location tokens), `billing_code` via `PROCEDURE_CODES`, `affordability_important` (phrases: uninsured, no insurance, low-cost, cheap, afford, sliding, free, self-pay, price_query), `require_verified_affordability` (regex `\bonly\b.*\b(verified|sliding[- ]fee|free)\b`), `location_required` (ZIP present or "in <city>"). Deterministic enrichment always wins over LLM output for these flags.

### 5.2 Retrieval (`rag/retrieve.py`)

Repository addition (both classes, same signature):

```python
def fetch_provider_corpus(self, category: str = "All") -> list[dict]: ...
```
Demo: full CSV (cached once per process, `functools.lru_cache`), category-filtered. Snowflake: `SELECT ... FROM CARE_AI.CURATED.TEMPE_CARE_OPTIONS WHERE (%s='All' OR CATEGORY=%s)` (parameterized; picks up verified programs). Prices: `fetch_price_rows(billing_code: str | None) -> list[dict]` with the same pattern.

```python
class Retriever:
    def __init__(self, repo, semantic: SemanticScorer | None = None, cfg=RETRIEVAL): ...
    def retrieve_providers(self, intent: CareIntentV2, top_k: int = 50) -> RetrievalResult
        # RetrievalResult(pool: list[ProviderRecord], scoped_out: dict[str,int], lexical_ranks, semantic_ranks)
```

Steps:
1. Scope: `fetch_provider_corpus(intent.category)` → `ingest.to_provider_records`. Category is pushed down as a retrieval scope (like today's `search`). The count removed by scoping is recorded as `scoped_out["wrong_category"]` (computed from a cached unscoped count) so zero-result messages can name it.
2. Query text = `keywords` + `SERVICE_SYNONYMS` expansions + procedure name. If the query text is empty (e.g. "a clinic near 85281" where the only term is the category), lexical/semantic scores are skipped and the scoped corpus is ordered by `(name asc)` before pooling — ranking later re-orders deterministically anyway.
3. Lexical: in-house BM25 over `name + specialty + category` (address excluded to avoid street-name noise), top 50.
4. Semantic — **decision**: no embedding model is available offline and adding one (sentence-transformers) is a heavy dependency for a hackathon repo. Local semantic = cosine similarity of character-trigram TF-IDF vectors between the synonym-expanded query and `name + specialty + category`. It is deterministic, pure Python, and catches morphological variants ("therapist"/"therapy", "pediatric"/"pediatrics"). Interface `SemanticScorer.score(query, docs) -> list[float]` allows a later `CortexEmbedScorer` (EMBED_TEXT_768 + VECTOR_COSINE_SIMILARITY) without changing callers; not implemented in this iteration.
5. RRF: `score(d) = Σ 1/(rrf_k + rank_i(d))` over both lists; ties broken by `source_id` asc for determinism.
6. Dedupe on `(npi, normalized_location)` where `normalized_location = normalize(address) or "zip:"+zip` (upper-case, collapse whitespace, strip punctuation, `STE/SUITE/#` → `STE`). Keep the first by RRF order; count duplicates as `filtered_out["duplicate_npi_location"]`.
7. `pool = fused[:40]`.

The semantic score per record (0–1) is carried forward because service-match "semantic only" uses it.

### 5.3 Eligibility filter (`rag/rank.py`)

```python
class EligibilityFilter:
    def apply(self, candidates: list[ProviderRecord], intent: CareIntentV2,
              price_index: PriceIndex) -> FilterResult  # (eligible, filtered_out: dict[str,int])
```

Hard filters, applied in this order, each counted:

| Filter key | Rule |
|---|---|
| `inactive` | `active_state == VERIFIED_NO` (unknown is kept, scored lower in confidence) |
| `invalid_record` | missing NPI, NPI not 10 digits or failing the NPI Luhn check (prefix 80840), empty name, empty category |
| `wrong_category` | `intent.category != "All"` and record category differs (re-check; normally 0 after scoping) |
| `outside_area` | only if `intent.location_required`: provider ZIP ≠ user ZIP **and** (no coordinates or distance > 20 mi) **and** city ≠ requested city. Since every demo row is Tempe, a Tempe ZIP request keeps Tempe-city rows in other ZIPs; they score lower on proximity rather than being dropped (see A3) |
| `missing_billing_procedure` | only if `intent.price_query and intent.billing_code` **and** `price_index.has_eligible_prices(code)`: drop providers with no price-eligible row for that exact code. If no price-eligible rows exist for the code (true today), the filter is **not applied** and a limitation is added (see D2) |
| `duplicate_npi_location` | second safeguard after retrieval dedupe |
| `affordability_required` | only if `intent.require_verified_affordability`: drop unless `affordability` ∈ verified positive states that satisfy `required_affordability` (sliding fee requested → SLIDING_FEE or FREE_PROGRAM accepted) |

Missing address, phone, specialty, or stale `last_updated` never filter.

### 5.4 Component scoring (`rag/rank.py`)

```python
def service_match_score(rec: ProviderRecord, intent: CareIntentV2, semantic_sim: float) -> tuple[float, ScoredFact]
def proximity(rec, user_loc: UserLocation | None) -> tuple[float | None, float | None, str | None, ScoredFact]
def proximity_score(distance_miles: float | None) -> float    # spec function, verbatim thresholds
def affordability_score(rec) -> tuple[float, AffordabilityEvidence, ScoredFact]
def price_score(rec, price_index: PriceIndex, intent) -> tuple[float | None, ScoredFact]
def financial_score(rec, intent, price_index) -> tuple[float | None, str, ScoredFact]
def confidence_score(rec, today: date) -> tuple[float, ScoredFact]
def availability_score(rec) -> tuple[float | None, ScoredFact]
def weighted_score(scores: dict[str, float | None], weights: RankingWeights) -> float
```

Service match (deterministic tiers, first matching tier wins):

| Tier | Rule | Score |
|---|---|---|
| exact | a query term maps via `SPECIALTY_TERMS` to the record's exact specialty, or record has a price-eligible row for `intent.billing_code` | 100 |
| category + strong keyword in specialty | category matches and a keyword/synonym occurs in `specialty` | 90 |
| category + strong keyword in name | category matches and keyword/synonym occurs in `name` | 85 |
| category only | category matches (or intent category was inferred and record has it), no keyword hit | 70 |
| semantic only | category mismatch/"All", `semantic_sim > 0` | `40 + 29 * semantic_sim` (40–69) |
| weak | none of the above | `39 * semantic_sim` (< 40) |

If the intent has no service terms and category is "All", every record gets 50 (`evidence_state=NOT_APPLICABLE`); it is constant across candidates so it cannot affect order.

Proximity:
- `UserLocation` is built from the intent: ZIP and/or city; coordinates only when a ZIP-centroid table is present (below). It gets internal source ID `USER_LOCATION:geocode:request`, which is never a citation.
- If the user gave no location → dimension `None` (not applicable) → weight renormalized away; `distance_method=None`.
- Otherwise the fallback chain, first available wins:
  1. `provider_coordinates`: both user and provider have real coordinates → haversine miles.
  2. `zip_centroid`: centroid coordinates for both ZIPs from `data/zip_centroids.csv` (optional; columns `zip,latitude,longitude,source`; must come from the Census ZCTA Gazetteer, not hand-entered). Distance is reported as approximate.
  3. `same_zip`: ZIP equal → `distance_miles=None`.
  4. `same_city`: city equal → `distance_miles=None`.
  5. `unknown` → `distance_miles=None`.
- Score = `proximity_score(distance_miles)`; any `None` distance → 40. No coordinates are invented; neither file exists today, so all current results use methods 3–5 and score 40 (A4).
- Snowflake option documented but not built: add `LATITUDE/LONGITUDE` to the curated table and compute `ST_DISTANCE(ST_MAKEPOINT(lon,lat), ...)/1609.344`.

Financial (`price_score` and `affordability_score` kept separate):
- `affordability_score`: table from spec via `AffordabilityEvidence` (100/90/80/60/30/0). Source: `VERIFIED_ACCESS_PROGRAMS.program_type` mapped by `ingest.map_program_type` (`"sliding fee"`→SLIDING_FEE, `"free"`/`"charity"`→FREE_PROGRAM, `"discount"`→DISCOUNTED_SELF_PAY). The NPPES default string maps to UNKNOWN. Category is never used to infer affordability.
- `price_score`: only over matched comparable records (§5.5). `100*(max-p)/(max-min)`; all equal → 100; provider without a matched price → `None`.
- `financial_score` selection:
  - `intent.price_query` → `price_score`.
  - else `intent.affordability_important` → `affordability_score`.
  - else → `None` (not applicable, renormalized away).
- If `affordability_important` (which includes price queries) and the selected financial score is `None` → 30 with `financial_evidence_state="unknown"`. This substitution is the one exception to renormalization.

Confidence: `+35 active_state==VERIFIED_YES`, `+20 address non-empty and zip non-empty`, `+15 phone`, `+15 specialty non-empty`, `+15 last_updated within 2 years of today`. Range 0–100.

Availability: `100` only when `availability_state == VERIFIED_YES` from an authoritative source with a source ID; `0` when `VERIFIED_NO`; otherwise `None` (renormalized). No source exists today, so it is always `None`.

### 5.5 Price matching (`PriceIndex` in `rag/rank.py`)

```python
class PriceIndex:
    def __init__(self, rows: list[PriceRecord]): ...
    def has_eligible_prices(self, billing_code: str) -> bool
    def comparable_group(self, rec: ProviderRecord, intent) -> list[PriceRecord]
    @staticmethod
    def comparable(a: PriceRecord, b: PriceRecord) -> bool
```
Comparable = same `billing_code`, same `billing_code_type`, same `setting` (None == None allowed), same `rate_type`, snapshot dates within 365 days, and known location identity (`ccn` non-empty and linked to a provider NPI). `rate_type="cash"` is preferred when the user asks about self-pay/cash; otherwise `negotiated` rows are excluded from a cash comparison and vice versa (mixing types is never allowed).

`price_eligible = state in LOCAL_PRICE_STATES and npi is not None`. Every current row is CCN prefix `05` → `state="CA"` → `price_eligible=False`, so `price_score=None` for every Tempe provider and the limitation "Hospital price data in this dataset covers California hospitals and was not used to rank Tempe providers." is added whenever `intent.price_query`.

### 5.6 Weighted total, renormalization, tie-break

```python
def weighted_score(scores, weights=WEIGHTS):
    available = {n: s for n, s in scores.items() if s is not None}
    w = sum(getattr(weights, n) for n in available)
    return 0.0 if w == 0 else round(sum(getattr(weights, n) * s for n, s in available.items()) / w, 4)
```
Renormalization: a `None` component is dropped and the remaining weights are divided by their sum, so a provider is never penalized with an implicit zero. Financial is pre-substituted with 30 when `affordability_important` (so it stays in the denominator). `ranking_basis` reports the effective weights for the request, e.g. `["service match 50%", "data confidence 14%", "financial 29%", "proximity: not applicable (no location given)", "availability: not verified for any provider"]`. Because dimension availability is decided per request (not per provider) for proximity and financial, all candidates in one request share the same denominator; availability is per provider but `None` for all today (A5).

Total rounded to 4 decimals before sorting so float noise cannot change order.

Sort key (ascending tuple):
```python
(-total, -service_match, distance_miles if distance_miles is not None else inf,
 -FINANCIAL_PRIORITY[financial_evidence_state], -confidence, name.casefold(), source_id)
```
`FINANCIAL_PRIORITY`: free 5, sliding 4, discounted 3, cash-only 2, unknown 1, no-assistance 0. `source_id` is a final determinism guard beyond the spec's six keys.

### 5.7 Selection and count rules

`selected = ranked[:3]`; ranks assigned `["A","B","C"][:len(selected)]` in code.
- ≥3 eligible → exactly 3.
- 2 → A, B + limitation "Only two providers met the requirements for this request."
- 1 → A + "Only one provider met ...".
- 0 → no LLM call. `limitations` names the constraint(s) that eliminated candidates: the largest non-zero counter from `scoped_out`/`filtered_out` (ties: all listed), e.g. "No providers met the requirements. 1,777 behavioral-health records were in scope; all were removed by: affordability_required (verified sliding-fee programs only)."
- Never pad with lower-relevance records. Weak-tier providers (service < 40) are still eligible if they passed the filters; they appear only if fewer strong ones exist (A6).

### 5.8 ScoredFacts and source IDs (`rag/citations.py`)

```python
def provider_source_id(npi: str, address: str, zip_code: str) -> str
    # "NPPES:<npi>:address:<sha1(normalize(address)|zip)[:10]>"
def price_source_id(row) -> str      # "HPT:<ccn>:<code_type>:<code>:<rate_type>:<sha1(payer|amount|snapshot)[:10]>"
def program_source_id(npi, address) -> str   # "VAP:<npi>:<hash>"
USER_LOCATION_ID = "USER_LOCATION:geocode:request"
def allowed_citations(selected: list[RankedProvider]) -> set[str]   # union of fact source_ids, minus USER_LOCATION_ID
def build_facts(rec, breakdown, intent) -> list[ScoredFact]
```
Each provider gets facts `service_match`, `proximity`, `financial`, `confidence`, `availability`, each with an `explanation` string written by code (e.g. "Specialty 'Mental Health' matches 'therapist' (category + specialty keyword)", "Same ZIP 85281 as request; exact distance unknown", "Price unknown: no comparable local price record", "Affordability not stated in NPPES"). `unknown_facts` = facts with `evidence_state == UNKNOWN`. Hashes are stable across runs (no salts, no timestamps).

## 6. Explanation generation

### 6.1 Cortex client (`llm/cortex_messages.py`)

```python
class CortexMessagesClient:
    def __init__(self, host: str, pat: str, model: str | None = None, timeout: float = 30.0): ...
        # model default: env CORTEX_MESSAGES_MODEL or "claude-sonnet-4-5"
    @classmethod
    def from_secrets(cls) -> "CortexMessagesClient | None"   # uses repository._load_snowflake_secrets(); None if host/api_key absent
    def generate_json(self, system: str, user: str, schema: dict, max_tokens: int = 1500) -> dict
```
- `POST https://{host}/api/v2/cortex/v1/messages` via `requests` (already a dependency; no `anthropic` SDK added). Headers: `Authorization: Bearer <PAT>`, `Content-Type: application/json`, `anthropic-version: 2023-06-01`, `X-Snowflake-Authorization-Token-Type: PROGRAMMATIC_ACCESS_TOKEN`.
- Body: `{"model", "max_tokens", "system", "messages":[{"role":"user","content":user}], "temperature": 0, "output_config": {"format": {"type": "json_schema", "schema": schema}}}`. The response JSON is `content[0].text`, parsed with `json.loads`.
- Raises `LLMUnavailable` on network error, non-200, or unparseable JSON. No retry loop beyond one retry on 429/503 with 1 s backoff. The PAT is never logged.
- Reference: Snowflake Cortex REST API docs (Messages API supports Claude models only; structured output via `output_config` json_schema).

Other backends implement the same `LLMClient` protocol (`generate_json`): `OllamaJSONClient` wraps `CareAgent.chat(messages, schema)`. The schema passed is `TopThreeExplanation.model_json_schema()` post-processed to add `additionalProperties: false` on every object and inline `$defs` (Messages API structured output requires a self-contained schema; assumption A7).

### 6.2 `ExplanationGenerator` (`rag/service.py`)

```python
class ExplanationGenerator:
    def __init__(self, llm: LLMClient | None): ...
    def explain_ranking(self, question, intent, providers: list[RankedProvider]) -> tuple[TopThreeExplanation, Literal["llm","template"]]
```
- User payload (JSON) per provider, as spec: `rank, name, total_score, ranking_reasons{service|proximity|financial|confidence|availability: {score, fact, source_id}}, unknowns`, plus `query_interpretation` hint and `ranking_basis`. No raw rows, no other candidates.
- System prompt: the spec prompt verbatim, plus "Return JSON matching the schema. Keep each provider's rank and source_id exactly as given."
- `llm is None` or `LLMUnavailable` → template.

### 6.3 Template fallback (`rag/citations.py`)

`template_explanation(intent, selected) -> TopThreeExplanation`, pure function of the ScoredFacts:
- heading: A "Highest-ranked match in the current dataset for the factors in this request", B "Second-ranked match", C "Third-ranked match".
- reason: top two contributing components by `weight*score`, phrased from fact explanations.
- `comparison_to_next`: names the first sort-key component where this provider beats the next (e.g. "Ranks above B on service match (90 vs 70)"; if totals tie, names the tie-break key).
- citations: the provider's fact source IDs.

## 7. Validation (`rag/validate.py`)

```python
class ExplanationValidationError(Exception): reasons: list[str]
class ExplanationValidator:
    def validate(self, explanation: TopThreeExplanation, selected: list[RankedProvider]) -> None   # raises
def check_claims(text: str, provider: RankedProvider) -> list[str]
```
Structural (spec): `1 ≤ len ≤ 3` and `len == len(selected)`; ranks `== ["A","B","C"][:n]`; per-position `source_id` equals selected source ID at that position (order, not just set equality — stricter than the spec, D3); `citation_source_ids ⊆ allowed_citations(selected)` and non-empty; `USER_LOCATION_ID` not cited.

Claim checks on all text fields (case-insensitive regex), each allowed only when the provider's evidence verifies it:
- `\d+(\.\d+)?\s*(mi|miles)` → only if `distance_method in {provider_coordinates, zip_centroid}`; the number must equal `distance_miles` rounded to 1 dp.
- accepting new patients / walk-in / open now / hours → only if `availability_state == VERIFIED_YES`.
- free / no cost / low-cost / sliding / discount → only if matching verified `AffordabilityEvidence`.
- expensive / more expensive / pricier / costly → only if `price_score` is not None.
- best provider / top-rated / highest quality / best doctor → always rejected.
- dollar amounts (`\$\d`) → only if a matched price fact exists and the amount appears in it.

Any failure → log reasons (no PII beyond provider names) and replace with `template_explanation`, `explanation_source="template"`. The template is itself run through the validator in tests to prove it passes.

## 8. Service (`rag/service.py`)

```python
class ProviderRecommendationService:
    def __init__(self, repo, intent_parser: IntentParser, retriever: Retriever,
                 eligibility: EligibilityFilter, ranker: Ranker,
                 generator: ExplanationGenerator, validator: ExplanationValidator,
                 today: Callable[[], date] = date.today): ...
    def recommend(self, question: str, conversation_context: list[dict] | None = None) -> TopProviderResponse

def build_recommendation_service(repo, ai_backend: str) -> ProviderRecommendationService
    # "Cortex (Snowflake)" -> CortexMessagesClient.from_secrets() (None -> template)
    # "Ollama (local)"     -> OllamaJSONClient(CareAgent())
    # anything else        -> llm=None
```
`recommend` follows §5 exactly; `conversation_context` is accepted and ignored in this iteration (spec signature). The price index is built once per service instance from `repo.fetch_price_rows()`. `today` is injectable for deterministic freshness tests.

`Ranker.rank(candidates, intent) -> list[RankedProvider]` returns all eligible providers sorted with ranks unset except the first three (ranks typed `Literal`; internal sorted list uses an internal `ScoredCandidate` dataclass and only the top 3 are converted to `RankedProvider`).

## 9. app.py integration (minimal, no visual change)

Changes are confined to the AI Care Guide button handler and imports:

```python
from rag.service import build_recommendation_service
...
@st.cache_resource
def get_recommendation_service(m, backend):
    return build_recommendation_service(get_repo(m), backend)
...
if st.button("Guide me", use_container_width=True):
    if not q.strip():
        st.warning("Describe what you need.")
    else:
        recommendation_service = get_recommendation_service(mode, ai_backend)
        result = recommendation_service.recommend(q)
        if result.emergency:
            st.error("Your message may describe an emergency. Call 911 now.")
        else:
            st.write(result.to_markdown())
```
- Same widgets, same `st.error` / `st.write` calls, same expander below; the rendered text block is the "Top 3 providers / A. Name — heading / Why: ... / Sources: ... / Limitations: ..." markdown from `to_markdown()`. The removed lines are the `agent.parse`, `explain`, `explain_prices`, and sidebar-row filtering inside this handler. `get_agent`, the sidebar, Directory and Price tabs are untouched.
- Retrieval is fresh from the repository; it no longer reuses `st.session_state.rows`.
- `to_markdown()` layout: `Top 3 providers` (or `Top 2`/`Top provider`), then for each provider `**A. Name — heading**`, `Why: reason`, bullet lines for service / proximity / price-affordability / unknowns / comparison, `Sources: id, id`; then `Limitations:` bullets and the ranking disclaimer.

## 10. Testing

`requirements-dev.txt`: `pytest==8.3.5` (exact pin; implementer confirms it installs on Python 3.11 with `uv pip install -r requirements-dev.txt`). Run: `uv run python -m pytest -q`. Tests use in-memory fixture records, fake LLM clients, and a fixed `today`; no network, no Snowflake, no secrets.

| File | Covers |
|---|---|
| `test_ranking.py` | weights load and sum to 1; `weighted_score` renormalizes `None`; `proximity_score` thresholds at 1/3/5/10/20 boundaries and None→40; service tiers (therapist/Behavioral health+Mental Health specialty → 90, generic clinic → semantic tier < 70); confidence arithmetic; full tie-break chain (each key decides one fixture pair); determinism (same input shuffled → same output) |
| `test_missing_data.py` | EvidenceState used instead of bools; unknown affordability → 30 when affordability important, `None` and renormalized when not; never 0 for unknown; unknown availability → None not 0; missing address/phone/specialty lowers confidence but is not filtered; `same_zip` → `distance_miles is None`, score 40; no-location request → proximity None |
| `test_top_three.py` | ≥3 → exactly 3 with A/B/C; 2 → A,B + limitation; 1 → A; 0 → elimination message names the constraint; no padding; emergency short-circuits before retrieval and LLM; "only verified sliding-fee clinics" becomes a hard filter; dedupe by NPI+normalized location; LLM returning reordered/extra/missing providers → template fallback with original order; LLM unavailable → template |
| `test_citations.py` | source-ID format and stability; citations ⊆ allowed set; USER_LOCATION never cited; claim checker rejects "2.4 miles" on same_zip, "accepting new patients"/"walk-in" without verification, "free"/"low-cost" without verified program, "more expensive" for unknown price, "best provider"; template explanation passes validator |
| `test_price_matching.py` | comparable requires same code, code type, setting, rate type, compatible dates, known identity; mixed rate types never compared; min-max formula; all-equal → 100; California CCN `05xxxx` → `price_eligible=False`, `price_score=None`; price query with only CA data skips `missing_billing_procedure` and adds the limitation; with AZ eligible fixtures the filter drops providers lacking the code |

Also run `uv run python -m py_compile app.py agent.py repository.py rag/*.py llm/*.py` and `uv run python scripts/check_data.py`.

## 11. Implementation order

1. `rag/config.py`, `rag/schemas.py`, `rag/citations.py` source IDs; `ingest.py`.
2. `repository.py` `fetch_provider_corpus` / `fetch_price_rows`.
3. `retrieve.py` (BM25, trigram semantic, RRF, dedupe).
4. `rank.py` (filter, scorers, PriceIndex, weighted_score, tie-break).
5. Template explanation + `validate.py`.
6. `llm/cortex_messages.py`, Ollama adapter, `ExplanationGenerator`.
7. `service.py` + factory; `TopProviderResponse.to_markdown`.
8. Tests (written alongside each step); `app.py` handler swap last.

## 12. Open assumptions (reviewer please check)

- A1 Active status: rows from `TEMPE_PROVIDERS` (curated with `IS_ACTIVE = TRUE`) are `active_state=VERIFIED_YES` with the provider source ID as evidence. If an `active_status`/`is_active` column is present it takes precedence; any other source → `UNKNOWN`.
- A2 "Invalid/incomplete record" means unusable identity (bad/missing NPI, empty name/category), not missing address/phone/specialty, because 80% of rows lack an address.
- A3 `outside_area`: the dataset is Tempe-only, so a ZIP request does not drop other Tempe ZIPs; only records outside the requested ZIP **and** city are removed. A user saying "only in 85281" sets a strict mode (`zip_strict=True` in intent) that drops other ZIPs.
- A4 No coordinates exist today; proximity will be 40 for every provider whenever location matters, so it cannot differentiate until `latitude/longitude` are exported or `data/zip_centroids.csv` (Census Gazetteer) is added. This is the spec's "add proximity only once coordinates are available" taken literally: the code path exists, the data does not.
- A5 Availability is `None` for everyone today (no authoritative source), so its 5% is always renormalized away.
- A6 Weak-match providers that pass hard filters may appear when fewer than 3 stronger ones exist; this is ranking, not padding, since they met every hard requirement. If the reviewer prefers, add a `min_service_score` (e.g. 40) as an extra hard filter in config.
- A7 Cortex Messages structured output accepts a JSON schema with `additionalProperties: false` and inlined `$defs`; the default model `claude-sonnet-4-5` is enabled in the account. If not, the client fails closed to the template.
- A8 The Cortex host comes from `.streamlit/secrets.toml [snowflake].host` and the PAT from `api_key`, the same values app.py already reads. Secrets are never logged or included in prompts.

## 13. Deviations from the spec (intentional)

- D1 `recommend()` returns `TopProviderResponse` (the spec's annotated return type) with the validated `TopThreeExplanation` embedded, rather than returning the explanation object as the spec's pseudo-code body does. app.py needs providers, limitations, and the emergency flag in one object.
- D2 The `missing_billing_procedure` hard filter only runs when price-eligible data exists for the requested code. Applied literally against the California-only sample it would remove every Tempe provider for every price question; instead the request proceeds with `financial=30 (unknown)` and a limitation.
- D3 Validation requires per-position source-ID equality, not only set equality, so a model that swaps B and C fails.
- D4 Service-match tier "category + strong keyword" is split into 90 (keyword in specialty) and 85 (keyword in name) to reproduce the spec's "therapist ≈ 90" example deterministically.
- D5 Added `source_id` as a 7th tie-break key so ordering is total even for identical names.
