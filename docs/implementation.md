# Tempe Care Compass: Current Implementation

How the app works today: file layout, data flow, and the retrieval + generation logic. Written as a baseline for designing a RAG upgrade.

## File structure
```
tempe-care-compass/
├── app.py                     # Streamlit UI, sidebar filters, 3 tabs, AI orchestration
├── agent.py                   # Intent parsing + LLM answer generation (Ollama / Cortex / fallback)
├── repository.py              # Data access: DemoRepository (CSV) / SnowflakeRepository (SQL)
├── .streamlit/secrets.toml    # [snowflake] host, account, user, api_key, role, warehouse, database, schema (gitignored)
├── .env                       # Optional: OLLAMA_URL, OLLAMA_MODEL, CORTEX_MODEL, SNOWFLAKE_* fallbacks (gitignored)
├── data/
│   ├── tempe_nppes_demo.csv   # 6,183 providers (exported from Snowflake)
│   ├── tempe_prices_demo.csv  # 1,000 hospital price rows (exported from Snowflake)
│   └── chat_history.json      # Chat log artifact
├── sql/01_curate_providers.sql# Snowflake pipeline (tables/views below)
├── scripts/
│   ├── export_data.py         # Snowflake -> data/*.csv snapshot refresh
│   └── check_data.py          # Data quality assertions on provider CSV
└── docs/                      # coco-run.md, demo-script.md, this file
```

## Data model

### Providers (`CARE_AI.CURATED.TEMPE_PROVIDERS`, dynamic table / `tempe_nppes_demo.csv`)
`npi, name, category, specialty, address, city, state, zip, phone, last_updated, affordability, source`
- Source: `AFFINE_NPPES_PROVIDER_DATA.REF_DW.DIM_PROVIDER`, filtered to active Tempe, AZ providers.
- `category` is one of: Clinic / primary care, Pharmacy, Dental, Behavioral health, Imaging, Hospital, Other care.
- `affordability` is "Unknown" unless matched to `VERIFIED_ACCESS_PROGRAMS`.

### Prices (`CARE_AI.CURATED.HOSPITAL_PRICES` view / `tempe_prices_demo.csv`)
`ccn, billing_code, billing_code_type, billing_code_description, payer_name, rate_type, rate_amount, snapshot_date`
- Source: `HEALTHPARSE_HOSPITAL_PRICE_TRANSPARENCY_RATES.HPT_NEGOTIATED_RATES_SAMPLE.HPT_NEGOTIATED_RATES` (California sample, few CPT codes).

### Other Snowflake objects
| Object | Purpose |
|---|---|
| `PRICE_COMPARISON` view | Per procedure: hospital_count, min/max/avg price, price_spread, min/max cash price |
| `VERIFIED_ACCESS_PROGRAMS` table | Manually verified sliding-fee / charity programs |
| `TEMPE_CARE_OPTIONS` view | Providers joined to verified programs |
| `CHAT_MESSAGES` table | Chat persistence (created, not wired into app.py) |

## Repository layer (`repository.py`)
Both classes expose the same interface:

| Method | Demo (pandas) | Snowflake (SQL) |
|---|---|---|
| `search(category, zip_code, query, limit)` | Filter by category, ZIP prefix, substring over name+specialty+address | `WHERE category = ? AND ZIP LIKE ?% AND (NAME/SPECIALTY ILIKE %?%)`, `ORDER BY NAME` |
| `search_prices(procedure, limit)` | Substring match on `billing_code_description` | `billing_code_description ILIKE %?%`, `ORDER BY description, rate_amount` |
| `get_price_comparison()` | groupby description, compute spread | `SELECT ... FROM PRICE_COMPARISON ORDER BY price_spread DESC LIMIT 50` |

Snowflake connection credential precedence: constructor args > `.streamlit/secrets.toml [snowflake]` > `SNOWFLAKE_*` env vars > defaults (`COMPUTE_WH`, `CARE_AI`, `CURATED`). `api_key` (PAT) is passed as `password`; `host` sets `port=443`. All SQL is parameterized.

Retrieval is purely lexical (exact substring / ILIKE). No ranking, embeddings, synonyms, or fuzzy matching.

## Agent layer (`agent.py`)

### `CareIntent` (pydantic)
`category="All", zip_code=None, emergency=False, keywords=[], price_query=False, procedure=None`

### Backends
| Class | parse() | generation |
|---|---|---|
| `CareAgent` (Ollama, `qwen3:8b`) | LLM structured output using `CareIntent.model_json_schema()`; falls back to `_fallback_parse` on error | `POST {OLLAMA_URL}/api/chat`, non-streaming, 45s timeout |
| `CortexAgent` (`llama3.1-8b`) | Always `_fallback_parse` (no LLM) | `SELECT SNOWFLAKE.CORTEX.COMPLETE(model, prompt)` over the existing connection |
| none | `_fallback_parse` | Template strings in app.py |

### `_fallback_parse(text)` (deterministic)
- Category: first keyword hit in an ordered dict (pharmacy → dental → behavioral → imaging → hospital → clinic).
- ZIP: regex `\b85\d{3}\b`.
- Emergency: phrases like "chest pain", "can't breathe", "overdose", "suicid".
- Price query: "cost", "price", "how much", "cheapest", "self-pay", etc.
- Procedure: first match in [mri, colonoscopy, x-ray, ct scan, knee replacement, knee arthroplasty].
- `keywords` is never populated.

### Generation ("augmented" step)
- `explain(question, rows)`: top 6 provider rows, projected to name/category/specialty/address/phone/affordability/last_updated, JSON-dumped into the prompt. Instructions: 2–3 leads, evidence only, no diagnosis, no "free" claims, call to verify, <170 words.
- `explain_prices(question, price_rows, provider_rows)`: top 8 price rows (description, payer, rate_type, amount, ccn) + up to 4 providers. Instructions: compare hospitals, highlight cheapest cash/self-pay, rates may vary, <200 words.
- Any LLM exception returns a templated list of names / rates.

## Request flow (`app.py`)
1. Load `.env`; read `HOST/ACCOUNT/USER/API_KEY/ROLE` from `st.secrets["snowflake"]` at import.
2. Sidebar: data source (Demo / Snowflake), AI backend (Cortex / Ollama / None), category, ZIP, keyword, limit, Search button.
3. `get_repo(mode)` (`@st.cache_resource`) builds the repository; Snowflake errors stop the app.
4. On first load or Search: `rows = repo.search(...)` stored in `st.session_state.rows`.
5. Tab 1, Directory: render provider cards + Google Maps link.
6. Tab 2, Prices: `get_price_comparison()` top 15 cards; free-text procedure box → `search_prices`.
7. Tab 3, AI Care Guide:
   ```
   q → agent.parse(q) (or _fallback_parse)
     → if emergency: show 911 banner
     → if price_query: price_rows = repo.search_prices(intent.procedure or "")
                       providers  = sidebar rows filtered by intent.category
                       answer     = agent.explain_prices(q, price_rows, providers[:4])
     → else:           providers  = sidebar rows filtered by intent.category (or all rows)
                       answer     = agent.explain(q, providers)
   ```
   Retrieval in the AI tab reuses the sidebar search results; it does not re-query with the parsed ZIP or keywords.

## Gaps relevant to a RAG redesign
- Retrieval is lexical and filter-based; questions phrased differently from column values miss ("therapist" vs "Behavioral health" works only via the hint list).
- Parsed `zip_code` and `keywords` are not used for retrieval in the AI tab.
- Context is a fixed top-N slice ordered by name, not by relevance.
- Cortex path has no LLM intent parsing.
- No chunking/embeddings; candidate options in Snowflake: `SNOWFLAKE.CORTEX.EMBED_TEXT_768` + `VECTOR_COSINE_SIMILARITY`, or Cortex Search Service over `TEMPE_PROVIDERS` / `HOSPITAL_PRICES` text.
- No conversation memory; `CHAT_MESSAGES` table exists but is unused.
- No citation of source rows (NPI / CCN) in answers.
- Price data covers California hospitals, so Tempe price answers are not local.
