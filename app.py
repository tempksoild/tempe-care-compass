# app.py — Streamlit UI for Tempe Care Compass
# Run: streamlit run app.py
# Renders three tabs: Provider Directory, Price Comparison, AI Care Guide.
# Sidebar selects data source (Demo/Snowflake), AI backend (Cortex/Ollama/None),
# care type, ZIP, keyword, and result limit.

import os
from pathlib import Path
from urllib.parse import quote_plus
import streamlit as st
from dotenv import load_dotenv
from agent import CareAgent, CortexAgent, _fallback_parse
from repository import DemoRepository, SnowflakeRepository

load_dotenv()
st.set_page_config(page_title="Tempe Care Compass", page_icon="✚", layout="wide")

CSS_FILE = Path(__file__).parent / "style.css"
if CSS_FILE.exists():
    st.html(CSS_FILE)

st.html(
    '<header class="hero">'
    "<small>PUBLIC-DATA CARE NAVIGATION</small>"
    "<h1>Tempe Care Compass</h1>"
    "<p>Find provider-directory leads and compare hospital prices before you call.</p>"
    "</header>"
    '<div class="warn"><b>Not medical advice.</b> Call 911 for a life-threatening emergency. '
    "Directory data does not prove affordability or availability. Published prices are not final costs.</div>"
)

# --- Sidebar ---
with st.sidebar:
    mode = st.radio("Data source", ["Demo snapshot", "Snowflake"])
    ai_backend = st.radio("AI backend", ["Cortex (Snowflake)", "Ollama (local)", "None"])
    st.divider()
    category = st.selectbox("Care type", ["All", "Clinic / primary care", "Pharmacy", "Dental", "Behavioral health", "Imaging", "Hospital", "Other care"])
    zipcode = st.text_input("ZIP code", placeholder="85281")
    keyword = st.text_input("Name or specialty", placeholder="urgent care")
    limit = st.slider("Results", 5, 50, 20)
    go = st.button("Search directory", type="primary", use_container_width=True)

    st.divider()
    with st.expander("Data Sources"):
        st.markdown(
            "**Provider Directory**\n"
            "- [Affine NPPES Provider Data](https://app.snowflake.com/marketplace/listing/GZT1Z2XIVUI) (Snowflake)\n"
            "- [CMS NPPES API](https://npiregistry.cms.hhs.gov/api-page)\n\n"
            "**Hospital Prices**\n"
            "- [Healthparse Transparency Rates](https://app.snowflake.com/marketplace/listing/GZT1Z4WB6KD) (Snowflake)\n"
            "- [CMS Price Transparency](https://www.cms.gov/priorities/key-initiatives/hospital-price-transparency)"
        )


@st.cache_resource
def get_repo(m):
    return SnowflakeRepository() if m == "Snowflake" else DemoRepository()


def get_agent(backend, repo_instance):
    if backend == "Cortex (Snowflake)" and hasattr(repo_instance, "con"):
        return CortexAgent(repo_instance.con)
    elif backend == "Ollama (local)":
        return CareAgent()
    return None


# --- Data fetch ---
r = get_repo(mode)
if "rows" not in st.session_state or go:
    try:
        st.session_state.rows = r.search(category, zipcode, keyword, limit)
        st.session_state.err = ""
    except Exception as e:
        st.session_state.rows = []
        st.session_state.err = str(e)

rows = st.session_state.rows

# --- Tabs ---
tab_dir, tab_prices, tab_ai = st.tabs(["Provider Directory", "Price Comparison", "AI Care Guide"])

# --- Tab 1: Provider Directory ---
with tab_dir:
    st.subheader(f"Directory results ({len(rows)})")
    if st.session_state.err:
        st.error(st.session_state.err)
    if not rows:
        st.info("No matches. Try removing the ZIP or choosing All.")
    for row in rows:
        addr = ", ".join(x for x in [row.get("address", ""), row.get("city", ""), row.get("state", ""), row.get("zip", "")] if x)
        maps = "https://www.google.com/maps/search/?api=1&query=" + quote_plus(addr)
        phone_display = row.get("phone", "") or "Phone not listed"
        st.html(
            f'<article class="card">'
            f'<span class="badge">{row.get("category", "Care")}</span>'
            f'<h3>{row.get("name", "Unnamed")}</h3>'
            f'<div class="meta">{row.get("specialty", "")}<br>{addr}<br>{phone_display}</div>'
            f'<small>Affordability: {row.get("affordability", "Unknown—call to verify")} &middot; Updated: {row.get("last_updated", "Unknown")}</small><br>'
            f'<a href="{maps}" target="_blank" rel="noopener noreferrer">Directions ↗</a>'
            f"</article>"
        )

# --- Tab 2: Price Comparison ---
with tab_prices:
    st.subheader("Hospital Price Transparency")
    st.caption("Published rates from CMS-mandated hospital price files (Healthparse Marketplace sample). Actual costs may vary.")

    comparisons = r.get_price_comparison()
    if not comparisons:
        st.info("No price comparison data available in this mode.")
    else:
        for comp in comparisons[:15]:
            desc = comp.get("billing_code_description", "Unknown procedure")
            min_p = comp.get("min_price", 0)
            max_p = comp.get("max_price", 0)
            avg_p = comp.get("avg_price", 0)
            spread = comp.get("price_spread", 0)
            n_hospitals = comp.get("hospital_count", 0)
            cash_min = comp.get("min_cash_price")
            cash_max = comp.get("max_cash_price")

            cash_line = ""
            if cash_min and cash_max and cash_min != cash_max:
                cash_line = f'<br>Cash/self-pay range: <span class="price-low">${cash_min:,.0f}</span> – <span class="price-high">${cash_max:,.0f}</span>'
            elif cash_min:
                cash_line = f'<br>Cash/self-pay: <span class="price-low">${cash_min:,.0f}</span>'

            st.html(
                f'<article class="card">'
                f"<h3>{desc}</h3>"
                f'<div class="meta">Across {n_hospitals} hospitals</div>'
                f'<span class="price-low">${min_p:,.0f}</span> – <span class="price-high">${max_p:,.0f}</span> '
                f"(avg ${avg_p:,.0f}, spread ${spread:,.0f})"
                f"{cash_line}"
                f"</article>"
            )

    st.divider()
    proc_search = st.text_input("Search for a procedure", placeholder="MRI, colonoscopy, knee...")
    if proc_search:
        price_results = r.search_prices(proc_search, limit=20)
        if price_results:
            for pr in price_results:
                st.html(
                    f'<article class="card">'
                    f'<span class="badge">{pr.get("rate_type", "rate")}</span> '
                    f'<span class="badge">{pr.get("billing_code_type", "")}: {pr.get("billing_code", "")}</span>'
                    f'<h3>{pr.get("billing_code_description", "")}</h3>'
                    f'<div class="meta">Hospital CCN: {pr.get("ccn", "N/A")} &middot; Payer: {pr.get("payer_name", "N/A")}</div>'
                    f'<b>${float(pr.get("rate_amount", 0)):,.2f}</b>'
                    f"</article>"
                )
        else:
            st.info("No price data found for that procedure.")

# --- Tab 3: AI Care Guide ---
with tab_ai:
    st.subheader("AI Care Guide")
    if ai_backend == "None":
        st.info("Select an AI backend in the sidebar to enable the care guide.")
    else:
        engine_label = "Snowflake Cortex" if "Cortex" in ai_backend else f"Ollama ({os.getenv('OLLAMA_MODEL', 'qwen3:8b')})"
        st.caption(f"Powered by {engine_label}. Fallback works if the model is unavailable.")

        q = st.text_area("What do you need?", placeholder="I need a low-cost walk-in clinic near 85281, or: How much does an MRI cost?", height=130)
        if st.button("Guide me", use_container_width=True):
            if not q.strip():
                st.warning("Describe what you need.")
            else:
                agent = get_agent(ai_backend, r)
                if agent is None:
                    intent = _fallback_parse(q)
                else:
                    intent = agent.parse(q)

                if intent.emergency:
                    st.error("Your message may describe an emergency. Call 911 now.")

                if intent.price_query:
                    price_results = r.search_prices(intent.procedure or "", limit=20)
                    filtered = [row for row in rows if intent.category == "All" or row.get("category") == intent.category]
                    if agent:
                        st.write(agent.explain_prices(q, price_results, filtered[:4]))
                    else:
                        if price_results:
                            rates = [f"${r.get('rate_amount', '?')} ({r.get('rate_type', '?')})" for r in price_results[:4]]
                            st.write(f"Published rates found: {', '.join(rates)}. Call each hospital to confirm your actual cost.")
                        else:
                            st.write("No price data found. Try searching the Price Comparison tab.")
                else:
                    filtered = [row for row in rows if intent.category == "All" or row.get("category") == intent.category]
                    if agent:
                        st.write(agent.explain(q, filtered or rows))
                    else:
                        leads = ", ".join(row.get("name", "Unnamed") for row in (filtered or rows)[:3])
                        st.write(f"Possible leads: {leads}. Call to verify price, eligibility, and hours.")

        with st.expander("Questions to ask when calling"):
            st.markdown(
                "- What is the self-pay price?\n"
                "- Do you offer a sliding-fee scale?\n"
                "- What documents are needed?\n"
                "- Are you accepting new patients?\n"
                "- Are labs or prescriptions included?"
            )
