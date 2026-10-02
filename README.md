# Tempe Care Compass

Open-source care navigation using **Snowflake CoCo**, **Affine NPPES Provider Data**, **Healthparse Hospital Price Transparency**, and **Snowflake Cortex AI**.

> Provider directory only—not medical advice. Call 911 for emergencies. NPPES does not prove that care is free, low-cost, open, or accepting patients. Published hospital prices are not final costs.

## Features
- Search 6,000+ Tempe provider records by care type, ZIP, name, or specialty.
- **Price comparison**: see the same MRI or colonoscopy priced differently across hospitals.
- Demo mode with checked-in CSV snapshots; Snowflake mode with parameterized SQL.
- **Dual AI backends**: Snowflake Cortex (cloud) or Ollama/Qwen3 (local), with deterministic fallback.
- Price-aware AI guide that answers "how much does an MRI cost?" with published hospital rates.
- Separate table for independently verified sliding-fee or charity-care programs.

## Folder structure
```
tempe-care-compass/
├── app.py                  # Streamlit UI entry point — run with: streamlit run app.py
├── agent.py                # AI chatbot logic (Ollama + Cortex backends, intent parsing, grounded responses)
├── repository.py           # Data access layer (Demo CSV + Snowflake SQL for providers and prices)
├── geocoding.py            # Address-to-coordinate geocoding service (Mapbox + Nominatim + Tempe centroids)
├── style.css               # Modern high-contrast UI stylesheet
├── .env.example            # Environment variable template (copy to .env and fill in credentials)
├── data/
│   ├── tempe_nppes_demo.csv    # 280-row NPPES provider snapshot for demo mode
│   └── tempe_prices_demo.csv   # 48-row hospital pricing snapshot for demo mode
├── sql/
│   └── 01_curate_providers.sql # Full Snowflake pipeline: dynamic table, views, pricing
├── scripts/
│   └── check_data.py           # Data quality assertions on the demo CSV
├── docs/
│   ├── coco-run.md             # CoCo execution evidence (query IDs, screenshots)
│   └── demo-script.md          # 7-step, 3-minute demo outline
├── coco-prompts.md             # 4-phase CoCo runbook (Discover > Plan > Build > Review)
├── .cortex/
│   └── skills/
│       └── tempe-care-data/
│           └── SKILL.md        # CoCo project skill (6 rules for NPPES curation)
├── README.md
└── LICENSE                     # MIT
```

### File descriptions

**`app.py`** — Main Streamlit application. Renders a 3-tab layout:
1. **Provider Directory** — search and browse provider cards with Google Maps links
2. **Price Comparison** — side-by-side hospital pricing for common procedures
3. **AI Care Guide** — chatbot that recommends the single most relevant provider and renders a pinpoint map via `st.map`

Execution: `streamlit run app.py`

**`agent.py`** — AI chatbot engine with two backend classes:
- `CareAgent` — calls Ollama's local `/api/chat` endpoint (Qwen3 by default)
- `CortexAgent` — calls `SNOWFLAKE.CORTEX.COMPLETE()` via the Snowflake connector
- `_fallback_parse()` — deterministic keyword-based intent parser (no AI required)

Each agent exposes `parse()` (extract intent), `explain()` (provider recommendations), and `explain_prices()` (cost comparison). The fallback fires automatically when AI is unavailable.

**`repository.py`** — Data access layer with two repository classes:
- `DemoRepository` — reads `data/tempe_nppes_demo.csv` and `data/tempe_prices_demo.csv` using pandas
- `SnowflakeRepository` — executes parameterized SQL against `CARE_AI.CURATED.*` views and tables

Both expose identical methods: `search()`, `search_prices()`, and `get_price_comparison()`.

**`scripts/check_data.py`** — Data quality assertions on the demo CSV. Validates required columns, no null NPIs, all rows are AZ, no duplicate NPI+address pairs, and no unverified "free" claims. Execution: `python scripts/check_data.py`

**`sql/01_curate_providers.sql`** — Complete Snowflake pipeline. Creates:
- `CARE_AI.CURATED.TEMPE_PROVIDERS` dynamic table (6,183 active Tempe providers)
- `CARE_AI.CURATED.HOSPITAL_PRICES` view (normalized pricing rates)
- `CARE_AI.CURATED.PRICE_COMPARISON` view (same-procedure price spreads)
- `CARE_AI.CURATED.TEMPE_CARE_OPTIONS` view (providers + verified access programs)
- `CARE_AI.CURATED.VERIFIED_ACCESS_PROGRAMS` table (manually verified sliding-fee/charity)

## Data sources
| Dataset | Source | Coverage |
|---|---|---|
| NPPES providers | Affine NPPES Provider Data (Snowflake Marketplace) | 9.8M US providers, weekly refresh |
| Hospital prices | Healthparse Hospital Price Transparency Rates (Snowflake Marketplace) | Sample: 47 hospitals, 3 CPT codes |
| Verified programs | Manual curation (HRSA, 211) | Tempe-area only |

## Run locally (demo mode)
This project uses [uv](https://docs.astral.sh/uv/) to manage Python and dependencies.

### 1. Install uv
macOS:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or: brew install uv
```
Windows (PowerShell):
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# or: winget install --id=astral-sh.uv -e
```

### 2. Create the virtual environment and install libraries
macOS:
```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install "streamlit>=1.40,<2" "pandas>=2.2,<3" "requests>=2.32,<3" \
  "snowflake-connector-python[pandas]>=3.12,<4" "pydantic>=2.9,<3" "python-dotenv>=1.0,<2"
cp .env.example .env
```
Windows (PowerShell):
```powershell
uv venv --python 3.11
.venv\Scripts\Activate.ps1
uv pip install "streamlit>=1.40,<2" "pandas>=2.2,<3" "requests>=2.32,<3" `
  "snowflake-connector-python[pandas]>=3.12,<4" "pydantic>=2.9,<3" "python-dotenv>=1.0,<2"
Copy-Item .env.example .env
```

| Library | Purpose |
|---|---|
| `streamlit` | Web UI |
| `pandas` | Demo CSV loading and data frames |
| `requests` | Ollama HTTP calls |
| `snowflake-connector-python[pandas]` | Snowflake queries and Cortex calls |
| `pydantic` | Intent/response models |
| `python-dotenv` | Loads `.env` credentials |

### 3. Run the app
```bash
uv run streamlit run app.py
```
Optional local AI:
```bash
ollama pull qwen3:8b
ollama serve
```
The deterministic fallback works when both Ollama and Cortex are unavailable.

## Snowflake setup
1. Install **Affine NPPES Provider Data** from Snowflake Marketplace.
2. Install **Healthparse Hospital Price Transparency Rates** from Snowflake Marketplace.
3. Run `sql/01_curate_providers.sql` in a Snowsight worksheet to create the CARE_AI database, dynamic table, views, and pricing pipeline.
4. Make sure the Snowflake libraries are installed in your uv environment (included in step 2 above). To add them on their own:
   ```bash
   uv pip install "snowflake-connector-python[pandas]>=3.12,<4" "python-dotenv>=1.0,<2"
   ```
5. Fill in `.env` with your Snowflake credentials:
   ```
   SNOWFLAKE_ACCOUNT=<org>-<account>
   SNOWFLAKE_USER=<user>
   SNOWFLAKE_PASSWORD=<password>
   ```
6. Verify the connection (works on macOS and Windows):
   ```bash
   uv run python -c "import os, snowflake.connector as sf; from dotenv import load_dotenv; load_dotenv(); c=sf.connect(account=os.environ['SNOWFLAKE_ACCOUNT'], user=os.environ['SNOWFLAKE_USER'], password=os.environ['SNOWFLAKE_PASSWORD']); print(c.cursor().execute('select current_version()').fetchone())"
   ```
7. Run `uv run streamlit run app.py` and select **Snowflake** as data source and **Cortex** as AI backend in the sidebar.

## Snowflake objects created
| Object | Type | Description |
|---|---|---|
| `CARE_AI.CURATED.TEMPE_PROVIDERS` | Dynamic table | 6,183 active Tempe providers from NPPES |
| `CARE_AI.CURATED.HOSPITAL_PRICES` | View | Normalized hospital pricing rates |
| `CARE_AI.CURATED.PRICE_COMPARISON` | View | Same-procedure price spread across hospitals |
| `CARE_AI.CURATED.TEMPE_CARE_OPTIONS` | View | Providers joined with verified access programs |
| `CARE_AI.CURATED.VERIFIED_ACCESS_PROGRAMS` | Table | Manually verified sliding-fee/charity programs |

## Important design choices
- **NPPES cannot establish affordability.** The app never labels a provider as free without verification from `VERIFIED_ACCESS_PROGRAMS`.
- **Hospital pricing is a sample.** The Healthparse listing covers ~47 California hospitals and 3 CPT codes. Full national coverage requires the paid listing.
- **Cortex AI is optional.** The app works without any AI backend using deterministic keyword matching and fallback responses.

## Tests
```bash
uv run python scripts/check_data.py
uv run python -m py_compile app.py agent.py repository.py
```

## Sources
- Affine NPPES: Snowflake Marketplace listing `GZT1Z2XIVUI`
- Healthparse HPT: Snowflake Marketplace listing `GZT1Z4WB6KD` (NOTE: this dataset only labels the market price across California , no clinic options avaliable across Tempe )
- NPPES API: https://npiregistry.cms.hhs.gov/api-page
- HRSA health centers: https://data.hrsa.gov/topics/health-centers
- Hospital prices: https://www.cms.gov/priorities/key-initiatives/hospital-price-transparency
