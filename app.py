# app.py — Streamlit UI for Tempe Care Compass
# Run: streamlit run app.py
# Renders three tabs: AI Care Guide, Provider Directory, Price Comparison.
# Sidebar selects data source (Demo/Snowflake), AI backend (Cortex/Ollama/None),
# care type, ZIP, keyword, and result limit.

import os
from pathlib import Path
from urllib.parse import quote_plus
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
try:
    from agent import (
        CareAgent,
        CortexAgent,
        _fallback_parse,
        _fallback_explain_one,
        select_best_provider,
    )
except ImportError:
    from agent import CareAgent, CortexAgent, _fallback_parse

    def _fallback_explain_one(question: str, provider: dict) -> str:
        if not provider:
            return "No matching directory records were found. Broaden your search or call 211 for resource navigation."
        name = provider.get("name", "Unnamed Provider")
        specialty = provider.get("specialty", "general care")
        category = provider.get("category", "Care")
        addr = provider.get("address", "Tempe, AZ")
        phone = provider.get("phone") or "their office"
        aff = provider.get("affordability", "Not stated in NPPES — call to verify")
        return (
            f"The most relevant choice for your inquiry is **{name}** ({category} &middot; {specialty}), located at {addr}. "
            f"Affordability status: {aff}. "
            f"Please call {phone} prior to visiting to confirm pricing, accepted coverage, operating hours, and new-patient availability."
        )

    def select_best_provider(question: str, providers: list[dict], intent=None) -> dict | None:
        if not providers:
            return None
        return providers[0]

from geocoding import geocode_address
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
    ai_backend = st.radio("AI backend", ["Cortex (Snowflake)"])
    st.divider()
    category = st.selectbox("Care type", ["All", "Clinic / primary care", "Pharmacy", "Dental", "Behavioral health", "Imaging", "Hospital", "Other care"])
    zipcode = st.text_input("ZIP code", placeholder="85281")
    keyword = st.text_input("Name or specialty", placeholder="urgent care")
    limit = st.slider("Results", 5, 50, 20)
    go = st.button("Search directory", type="primary", use_container_width=True)


@st.cache_resource
def get_repo(m):
    return SnowflakeRepository() if m == "Snowflake" else DemoRepository()


def get_agent(backend, repo_instance):
    if backend == "Cortex (Snowflake)" and hasattr(repo_instance, "con"):
        return CortexAgent(repo_instance.con)
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
tab_ai, tab_dir, tab_prices = st.tabs(["AI Care Guide", "Provider Directory", "Price Comparison"])

# --- Tab 1: AI Care Guide ---
with tab_ai:
    st.subheader("AI Care Guide")
    st.caption("Powered by Snowflake Cortex. Fallback works if the model is unavailable.")

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
                candidate_category = intent.category if intent.category != "All" else ("Imaging" if any(w in q.lower() for w in ["mri", "x-ray", "imaging", "ct scan", "ultrasound"]) else "Hospital")
                proc_providers = r.search(category=candidate_category, zip_code=intent.zip_code or "", query=intent.procedure or "", limit=30)
                if not proc_providers:
                    proc_providers = [row for row in rows if intent.category == "All" or row.get("category") == intent.category] or rows

                if agent:
                    st.write(agent.explain_prices(q, price_results, proc_providers[:4]))
                else:
                    if price_results:
                        rates = [f"${r_item.get('rate_amount', '?')} ({r_item.get('rate_type', '?')})" for r_item in price_results[:4]]
                        st.write(f"Published rates found: {', '.join(rates)}. Call each hospital to confirm your actual cost.")
                    else:
                        st.write("No price data found. Try searching the Price Comparison tab.")

                best_provider = select_best_provider(q, proc_providers, intent=intent)
                if best_provider:
                    st.divider()
                    st.subheader("Recommended Provider Facility")
                    addr = ", ".join(x for x in [best_provider.get("address", ""), best_provider.get("city", ""), best_provider.get("state", ""), best_provider.get("zip", "")] if x)
                    maps = "https://www.google.com/maps/search/?api=1&query=" + quote_plus(addr)
                    phone_display = best_provider.get("phone", "") or "Phone not listed"

                    st.html(
                        f'<article class="card" style="border: 2px solid var(--teal-primary);">'
                        f'<span class="badge">{best_provider.get("category", "Care")}</span> '
                        f'<span class="badge" style="background:#e8f4f1; color:#075a55;">⭐ Recommended Facility</span>'
                        f'<h3>{best_provider.get("name", "Unnamed")}</h3>'
                        f'<div class="meta">{best_provider.get("specialty", "")}<br>{addr}<br>📞 {phone_display}</div>'
                        f'<small>Affordability: {best_provider.get("affordability", "Unknown — call to verify")} &middot; Updated: {best_provider.get("last_updated", "Unknown")}</small><br>'
                        f'<a href="{maps}" target="_blank" rel="noopener noreferrer">Directions ↗</a>'
                        f'</article>'
                    )

                    coords = geocode_address(
                        address=best_provider.get("address", ""),
                        city=best_provider.get("city", "Tempe"),
                        state=best_provider.get("state", "AZ"),
                        zip_code=best_provider.get("zip", ""),
                    )
                    map_df = pd.DataFrame([{
                        "latitude": coords["latitude"],
                        "longitude": coords["longitude"],
                    }])
                    st.subheader("Location Pinpoint")
                    st.caption(f"📍 {coords['formatted_address']} &middot; Coordinates: {coords['latitude']:.4f}° N, {abs(coords['longitude']):.4f}° W (via {coords['source']})")
                    st.map(map_df, latitude="latitude", longitude="longitude", zoom=14, color="#075a55")

            else:
                # Provider search inquiry
                candidate_rows = r.search(
                    category=intent.category,
                    zip_code=intent.zip_code or "",
                    query=intent.procedure or (" ".join(intent.keywords) if intent.keywords else ""),
                    limit=50,
                )
                if not candidate_rows:
                    candidate_rows = [row for row in rows if intent.category == "All" or row.get("category") == intent.category] or rows

                best_provider = select_best_provider(q, candidate_rows, intent=intent)

                if not best_provider:
                    st.info("No matching directory records were found. Broaden your search or call 211 for resource navigation.")
                else:
                    if agent and hasattr(agent, "explain_one"):
                        explanation = agent.explain_one(q, best_provider)
                    else:
                        explanation = _fallback_explain_one(q, best_provider)

                    st.subheader("Top Recommendation")
                    st.write(explanation)

                    addr = ", ".join(x for x in [best_provider.get("address", ""), best_provider.get("city", ""), best_provider.get("state", ""), best_provider.get("zip", "")] if x)
                    maps = "https://www.google.com/maps/search/?api=1&query=" + quote_plus(addr)
                    phone_display = best_provider.get("phone", "") or "Phone not listed"

                    st.html(
                        f'<article class="card" style="border: 2px solid var(--teal-primary);">'
                        f'<span class="badge">{best_provider.get("category", "Care")}</span> '
                        f'<span class="badge" style="background:#e8f4f1; color:#075a55;">⭐ Most Relevant Choice</span>'
                        f'<h3>{best_provider.get("name", "Unnamed")}</h3>'
                        f'<div class="meta">{best_provider.get("specialty", "")}<br>{addr}<br>📞 {phone_display}</div>'
                        f'<small>Affordability: {best_provider.get("affordability", "Unknown — call to verify")} &middot; Updated: {best_provider.get("last_updated", "Unknown")}</small><br>'
                        f'<a href="{maps}" target="_blank" rel="noopener noreferrer">Directions ↗</a>'
                        f'</article>'
                    )

                    coords = geocode_address(
                        address=best_provider.get("address", ""),
                        city=best_provider.get("city", "Tempe"),
                        state=best_provider.get("state", "AZ"),
                        zip_code=best_provider.get("zip", ""),
                    )
                    map_df = pd.DataFrame([{
                        "latitude": coords["latitude"],
                        "longitude": coords["longitude"],
                    }])
                    st.subheader("Location Pinpoint")
                    st.caption(f"📍 {coords['formatted_address']} &middot; Coordinates: {coords['latitude']:.4f}° N, {abs(coords['longitude']):.4f}° W (via {coords['source']})")
                    st.map(map_df, latitude="latitude", longitude="longitude", zoom=14, color="#075a55")

    with st.expander("Questions to ask when calling"):
        st.markdown(
            "- What is the self-pay price?\n"
            "- Do you offer a sliding-fee scale?\n"
            "- What documents are needed?\n"
            "- Are you accepting new patients?\n"
            "- Are labs or prescriptions included?"
        )

# --- Tab 2: Provider Directory ---
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

# --- Tab 3: Price Comparison ---
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
