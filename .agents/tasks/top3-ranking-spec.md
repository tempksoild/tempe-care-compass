# Spec: Deterministic Top-3 Provider Ranking (backend only)

Central rule: Retrieval finds candidates, deterministic code ranks them, and the LLM only explains why A, B and C received those positions. NO UI changes. `app.py` only swaps its orchestration call to `result = recommendation_service.recommend(q)`.

## Output shape
```
Top 3 providers
A. Provider name — Best overall
Why: Best combination of proximity, verified price or affordability, specialty match, and record confidence.
B. Provider name — Second best
Why: Strong specialty and proximity match, but price is unknown.
C. Provider name — Third best
Why: Relevant service, but farther away or less affordability information.
```
Per provider the explanation covers: proximity, price/affordability, service match, important unknowns, "why X ranks above next", sources. Say "highest-ranked match in the current dataset for the factors in this request", never "best provider" (no clinical-quality implication).

## Flow
User question → safety detection → structured intent parsing → retrieve all eligible candidates (fresh repository retrieval, not sidebar rows) → hard filters → scoring → exactly top 3 → send 3 records + score explanations to Cortex Messages API → A/B/C explanations → validate claims & source IDs → structured result to app.py.

## Eligibility (hard filters)
Remove: inactive providers; wrong requested category; outside requested area (if location required); missing exact billing procedure when price-related; duplicate NPI/normalized-location; invalid/incomplete records. Affordability is a ranking factor, except when user explicitly says e.g. "only verified sliding-fee clinics" → hard filter.

## Ranking weights (store in config, tunable)
service 0.35, proximity 0.30, financial 0.20, confidence 0.10, availability 0.05. Components 0–100. Unavailable dimensions → renormalize available weights (no zero substitution), EXCEPT financial: if affordability is important and financial is None, use neutral 30 (do not renormalize it away).

```python
def weighted_score(scores):
    available = {n: s for n, s in scores.items() if s is not None}
    w = sum(WEIGHTS[n] for n in available)
    return 0.0 if w == 0 else sum(WEIGHTS[n]*s for n, s in available.items())/w
```

### Service match
Exact billing code or specialty 100; exact category + strong keyword 85; category only 70; semantic only 40–69; weak/general <40. (e.g., "therapist" vs Behavioral Health ≈ 90; generic clinic ≈ 45.)

### Proximity
Needs coordinates for both user and provider; ZIP equality is a location match, not a distance.
```python
def proximity_score(d):
    if d is None: return 40
    if d <= 1: return 100
    if d <= 3: return 85
    if d <= 5: return 70
    if d <= 10: return 45
    if d <= 20: return 20
    return 0
```
Fallback order: exact address coords → ZIP centroid → same ZIP → same city → unknown. Expose `distance_miles` and `distance_method`. Never state "X miles" when only ZIP match is known. For local CSV, export latitude/longitude and compute haversine locally (Snowflake option: ST_DISTANCE on GEOGRAPHY returns meters). Add proximity only once coordinates are available.

### Price / affordability
Separate `price_score` and `affordability_score` internally; uninsured query → financial = affordability; price query (e.g., MRI) → financial = price.
Affordability: verified free program for matching eligibility 100; verified sliding fee 90; verified discounted self-pay 80; verified cash price, no assistance 60; unknown 30; evidence of no assistance 0. Unknown is described as "unknown", never "more expensive".
Price comparison only when same billing code, code type, care setting, price type, compatible dates, known hospital/location identity:
`price_score = 100*(max-p)/(max-min)`; all equal → 100.
If price data is still the California sample: `price_eligible=False, price_score=None`; do not use for Tempe ranking.

### Data confidence
+35 active status verified, +20 full address, +15 phone, +15 specialty, +15 updated within 2 years.

### Availability
Only score if an authoritative source explicitly verifies accepting new patients / walk-in / current hours. Otherwise None (not false).

## Missing-data policy
```python
class EvidenceState(str, Enum):
    VERIFIED_YES="verified_yes"; VERIFIED_NO="verified_no"; UNKNOWN="unknown"; NOT_APPLICABLE="not_applicable"
```

## Models (pydantic)
```python
class ScoredFact(BaseModel):
    name: str; score: float; explanation: str; source_ids: list[str]; evidence_state: EvidenceState
class RankingBreakdown(BaseModel):
    service_match: float; proximity: float|None; financial: float|None; confidence: float
    availability: float|None; total: float; distance_miles: float|None; distance_method: str|None
    financial_evidence_state: str; availability_evidence_state: str
class RankedProvider(BaseModel):
    rank: Literal["A","B","C"]; source_id: str; npi: str; name: str; category: str; specialty: str
    address: str; phone: str|None; scores: RankingBreakdown; evidence_facts: list[str]; unknown_facts: list[str]
class TopProviderResponse(BaseModel):
    query_understood_as: str; ranking_basis: list[str]; providers: list[RankedProvider]; limitations: list[str]
class ProviderExplanation(BaseModel):
    rank: Literal["A","B","C"]; source_id: str; heading: str; reason: str; proximity_explanation: str
    financial_explanation: str; service_explanation: str; important_unknowns: list[str]
    comparison_to_next: str|None; citation_source_ids: list[str]
class TopThreeExplanation(BaseModel):
    query_interpretation: str; providers: list[ProviderExplanation]; ranking_disclaimer: str
```
Source IDs like `NPPES:<npi>:address:<hash>`; user location recorded internally as `USER_LOCATION:geocode:request` (not a public citation).

## Tie-breaking
1 total desc, 2 service desc, 3 verified distance asc (None=inf), 4 financial evidence priority desc, 5 confidence desc, 6 name asc.

## Count rules
≥3 eligible → exactly 3; 2 → A,B + note only two met requirements; 1 → A only; 0 → say none met and which constraint eliminated them. Never pad with irrelevant providers.

## Retrieval depth
Lexical top 50 + semantic top 50 → reciprocal rank fusion → pool[:40] → hard filters → rank → top 3. Only top 3 + scoring evidence go to the LLM.

## Generation (Cortex Messages API, structured output via `output_config` JSON schema)
Prepared record per provider: rank, name, total_score, ranking_reasons{service/proximity/financial: score, fact, source_id}, unknowns.
System prompt:
```
You are explaining an already-calculated provider ranking.
Do not select, reorder, remove or add providers.
A, B and C were ranked by deterministic application logic.
Use only the supplied ranking facts.
Describe A as the highest-ranked match in the current dataset, not as the highest-quality medical provider.
For every provider: 1. service match 2. proximity only if distance verified 3. price or affordability 4. important unknowns 5. cite only supplied source IDs.
Never infer affordability from provider category. Never claim an unknown price is expensive.
Never claim accepting patients, walk-in, open, free or low-cost unless supplied evidence explicitly verifies it.
```
Validation: 1 ≤ len ≤ 3; ranks == ["A","B","C"][:n]; source_id set equals selected set; citations ⊆ supplied IDs. On failure, fall back to a deterministic template explanation built from the ScoredFacts.

## Service interface
```python
class ProviderRecommendationService:
    def recommend(self, question, conversation_context=None) -> TopProviderResponse:
        intent = self.intent_parser.parse(question)
        if intent.emergency: return self.emergency_response()
        candidates = self.retriever.retrieve_providers(intent=intent, top_k=50)
        eligible = self.filter.apply(candidates=candidates, intent=intent)
        ranked = self.ranker.rank(candidates=eligible, intent=intent)
        selected = ranked[:3]
        explanation = self.generator.explain_ranking(question=question, intent=intent, providers=selected)
        self.validator.validate(explanation=explanation, selected=selected)
        return explanation
```

## Files
```
rag/{schemas,ingest,retrieve,rank,citations,validate,service}.py
llm/cortex_messages.py
tests/{test_ranking,test_missing_data,test_top_three,test_citations,test_price_matching}.py
```

## Implementation order
1 stable source IDs on provider & price records; 2 intent triggers fresh repository retrieval; 3 retrieve 30–50 candidates; 4 dedupe NPI+normalized location; 5 deterministic scoring + tie-breaks; 6 A/B/C from code; 7 Cortex structured explanations; 8 validate order & source IDs; 9 proximity once coordinates exist; 10 financial only from verified affordability or comparable local prices.
