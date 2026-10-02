# agent.py — AI chatbot engine for Tempe Care Compass
# Provides two AI backend classes and a deterministic fallback:
#   CareAgent    — uses Ollama local LLM (default: qwen3:8b)
#   CortexAgent  — uses Snowflake Cortex COMPLETE() (default: llama3.1-8b)
#   _fallback_parse() — keyword-based intent parser, no AI required
#
# Each agent exposes:
#   parse(text)          -> CareIntent (category, zip, emergency, price_query)
#   explain(q, rows)     -> grounded provider recommendation (<=170 words)
#   explain_prices(q, price_rows, provider_rows) -> cost comparison (<=200 words)
#   explain_ranking(q, intent, providers) -> TopThreeExplanation for code-ranked A/B/C
#       Cortex: Messages API structured output; Ollama: /api/chat with JSON schema;
#       either falls back to rag.explain.template_explanation (no LLM).

import json, os, re, requests
from pydantic import BaseModel, Field

from rag import explain as rx
from rag import understand as ru
from rag.validate import validate_explanation


class CareIntent(BaseModel):
    category: str = "All"
    zip_code: str | None = None
    emergency: bool = False
    keywords: list[str] = Field(default_factory=list)
    price_query: bool = False
    procedure: str | None = None


class CareAgent:
    """Ollama-backed care navigation agent."""
    def __init__(self):
        self.url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "qwen3:8b")

    def chat(self, messages, schema=None):
        body = {"model": self.model, "messages": messages, "stream": False}
        if schema: body["format"] = schema
        r = requests.post(f"{self.url.rstrip('/')}/api/chat", json=body, timeout=45)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def parse(self, text):
        try:
            raw = self.chat([
                {"role":"system","content":"Extract directory filters. Categories: All, Clinic / primary care, Pharmacy, Dental, Behavioral health, Imaging, Hospital. If the user asks about prices, costs, or how much something costs, set price_query=true and extract the procedure name. Never diagnose."},
                {"role":"user","content":text}], CareIntent.model_json_schema())
            return CareIntent.model_validate_json(raw)
        except Exception:
            return _fallback_parse(text)

    def explain(self, question, rows):
        if not rows: return "No matching directory records were found. Broaden your search or call 211 for resource navigation."
        evidence = [{k:r.get(k,"") for k in ("name","category","specialty","address","phone","affordability","last_updated")} for r in rows[:6]]
        prompt = f"Question: {question}\nDirectory evidence: {json.dumps(evidence)}\nSuggest 2-3 directory leads using only the evidence. Do not diagnose or claim a provider is free unless explicitly verified. Tell the user to call to confirm price, eligibility, hours, and new-patient status. Under 170 words."
        try:
            return self.chat([{"role":"system","content":"You are a cautious healthcare-navigation assistant, not a clinician."},{"role":"user","content":prompt}])
        except Exception:
            return "Possible directory leads: " + ", ".join(r.get("name","Unnamed provider") for r in rows[:3]) + ". Call each location to verify price, eligibility, hours, and whether it accepts new patients."

    def explain_one(self, question, provider):
        if not provider:
            return "No matching directory records were found. Broaden your search or call 211 for resource navigation."
        evidence = {k: provider.get(k, "") for k in ("name", "category", "specialty", "address", "city", "state", "zip", "phone", "affordability", "last_updated")}
        prompt = (
            f"Question: {question}\n"
            f"Selected most relevant provider: {json.dumps(evidence)}\n"
            "Explain clearly why this single provider is the most relevant choice for the user's inquiry based on their specialty, location, and care type. "
            "Do not diagnose or claim a provider is free unless explicitly verified. "
            "Remind the user to call ahead to confirm price, accepted insurance, hours, and new-patient availability. Under 140 words."
        )
        try:
            return self.chat([{"role": "system", "content": "You are a cautious healthcare-navigation assistant, not a clinician."}, {"role": "user", "content": prompt}])
        except Exception:
            return _fallback_explain_one(question, provider)

    def explain_prices(self, question, price_rows, provider_rows=None):
        if not price_rows: return "No price data found for that procedure. Hospital pricing data is limited to the Healthparse sample."
        evidence = [{k: r.get(k, "") for k in ("billing_code_description", "payer_name", "rate_type", "rate_amount", "ccn")} for r in price_rows[:8]]
        providers = []
        if provider_rows:
            providers = [{k: r.get(k, "") for k in ("name", "category", "phone", "address")} for r in provider_rows[:4]]
        prompt = f"Question: {question}\nPrice evidence: {json.dumps(evidence)}\n"
        if providers:
            prompt += f"Nearby providers: {json.dumps(providers)}\n"
        prompt += "Compare prices across hospitals using only the evidence. Highlight the cheapest cash or self-pay option if available. Remind the user these are published rates and actual costs may vary. Under 200 words."
        try:
            return self.chat([{"role":"system","content":"You are a cautious healthcare cost navigation assistant. Never claim prices are final — they are published rates that may vary."},{"role":"user","content":prompt}])
        except Exception:
            rates = [f"${r.get('rate_amount','?')} ({r.get('rate_type','?')})" for r in price_rows[:4]]
            return f"Published rates found: {', '.join(rates)}. Call each hospital to confirm your actual cost."

    def understand_request(self, question):
        """LLM reads the whole request -> QueryPlan (care type, search terms, urgency, location)."""
        return _understand(lambda: self.chat([{"role": "system", "content": ru.UNDERSTAND_PROMPT},
                                              {"role": "user", "content": question}], ru.PLAN_SCHEMA),
                           self, question)

    def explain_ranking(self, question, intent, providers, empty_message=None):
        """Explain an order already computed by rag.rank; never re-rank."""
        return _explain_ranking(
            lambda: self.chat([{"role": "system", "content": rx.GENERATOR_PROMPT},
                               {"role": "user", "content": rx.user_message(question, intent, providers)}],
                              rx.EXPLANATION_SCHEMA),
            self, question, intent, providers, empty_message)


def _understand(call_llm, agent, question):
    """LLM structured plan, keyword fallback on any failure. Sets agent.last_understand_error."""
    agent.last_understand_error = None
    try:
        return ru.parse_plan(call_llm())
    except Exception as e:
        agent.last_understand_error = str(e)
        return ru.fallback_plan(question)


def _explain_ranking(call_llm, agent, question, intent, providers, empty_message):
    """Shared path: LLM structured output -> parse/guard -> template fallback.
    Sets agent.last_explain_source to 'llm' or 'template' (and last_error)."""
    agent.last_error = None
    if providers:
        try:
            exp = validate_explanation(rx.parse_explanation(call_llm(), providers), providers)
            agent.last_explain_source = "llm"
            return exp
        except Exception as e:  # LLM down, bad JSON, or ranking changed
            agent.last_error = str(e)
    agent.last_explain_source = "template"
    return rx.template_explanation(question, intent, providers, empty_message)


class CortexAgent:
    """Snowflake Cortex-backed care navigation agent."""
    def __init__(self, connection, messages_client=None):
        self.con = connection
        self.model = os.getenv("CORTEX_MODEL", "llama3.1-8b")
        self._messages = messages_client

    @property
    def messages(self):
        """Cortex Messages API client, built lazily from the existing connection."""
        if self._messages is None:
            from llm.cortex_messages import CortexMessagesClient
            self._messages = CortexMessagesClient.from_connection(self.con)
        return self._messages

    def understand_request(self, question):
        """Cortex Messages API reads the whole request -> QueryPlan."""
        return _understand(lambda: self.messages.create(ru.UNDERSTAND_PROMPT, question, ru.PLAN_SCHEMA,
                                                        max_tokens=600), self, question)

    def explain_ranking(self, question, intent, providers, empty_message=None):
        """Explain an order already computed by rag.rank; never re-rank."""
        return _explain_ranking(
            lambda: self.messages.create(rx.GENERATOR_PROMPT, rx.user_message(question, intent, providers),
                                         rx.EXPLANATION_SCHEMA),
            self, question, intent, providers, empty_message)

    def _complete(self, prompt):
        with self.con.cursor() as c:
            c.execute(
                "SELECT SNOWFLAKE.CORTEX.COMPLETE(%s, %s) AS response",
                (self.model, prompt),
            )
            return c.fetchone()[0]

    def parse(self, text):
        return _fallback_parse(text)

    def explain(self, question, rows):
        if not rows: return "No matching directory records were found. Broaden your search or call 211 for resource navigation."
        evidence = [{k: r.get(k, "") for k in ("name", "category", "specialty", "address", "phone", "affordability", "last_updated")} for r in rows[:6]]
        prompt = (
            "You are a cautious healthcare-navigation assistant, not a clinician.\n\n"
            f"Question: {question}\nDirectory evidence: {json.dumps(evidence)}\n"
            "Suggest 2-3 directory leads using only the evidence. Do not diagnose or claim a provider is free unless explicitly verified. "
            "Tell the user to call to confirm price, eligibility, hours, and new-patient status. Under 170 words."
        )
        try:
            return self._complete(prompt)
        except Exception:
            return "Possible directory leads: " + ", ".join(r.get("name", "Unnamed provider") for r in rows[:3]) + ". Call each location to verify price, eligibility, hours, and whether it accepts new patients."

    def explain_one(self, question, provider):
        if not provider:
            return "No matching directory records were found. Broaden your search or call 211 for resource navigation."
        evidence = {k: provider.get(k, "") for k in ("name", "category", "specialty", "address", "city", "state", "zip", "phone", "affordability", "last_updated")}
        prompt = (
            "You are a cautious healthcare-navigation assistant, not a clinician.\n\n"
            f"Question: {question}\n"
            f"Selected most relevant provider: {json.dumps(evidence)}\n"
            "Explain clearly why this single provider is the most relevant choice for the user's inquiry based on their specialty, location, and care type. "
            "Do not diagnose or claim a provider is free unless explicitly verified. "
            "Remind the user to call ahead to confirm price, accepted insurance, hours, and new-patient availability. Under 140 words."
        )
        try:
            return self._complete(prompt)
        except Exception:
            return _fallback_explain_one(question, provider)

    def explain_prices(self, question, price_rows, provider_rows=None):
        if not price_rows: return "No price data found for that procedure. Hospital pricing data is limited to the Healthparse sample."
        evidence = [{k: r.get(k, "") for k in ("billing_code_description", "payer_name", "rate_type", "rate_amount", "ccn")} for r in price_rows[:8]]
        providers = []
        if provider_rows:
            providers = [{k: r.get(k, "") for k in ("name", "category", "phone", "address")} for r in provider_rows[:4]]
        prompt = (
            "You are a cautious healthcare cost navigation assistant. Never claim prices are final.\n\n"
            f"Question: {question}\nPrice evidence: {json.dumps(evidence)}\n"
        )
        if providers:
            prompt += f"Nearby providers: {json.dumps(providers)}\n"
        prompt += "Compare prices across hospitals using only the evidence. Highlight the cheapest cash or self-pay option if available. Remind the user these are published rates and actual costs may vary. Under 200 words."
        try:
            return self._complete(prompt)
        except Exception:
            rates = [f"${r.get('rate_amount', '?')} ({r.get('rate_type', '?')})" for r in price_rows[:4]]
            return f"Published rates found: {', '.join(rates)}. Call each hospital to confirm your actual cost."


def _fallback_parse(text):
    low = text.lower()
    category = "All"
    hints = {
        "Pharmacy": ["pharmacy", "medicine", "prescription", "rx"],
        "Dental": ["dentist", "dental", "teeth"],
        "Behavioral health": ["mental", "therapy", "counselor", "anxiety", "depression"],
        "Imaging": ["mri", "x-ray", "imaging", "ct scan", "ultrasound"],
        "Hospital": ["hospital", "emergency", "er "],
        "Clinic / primary care": ["clinic", "doctor", "urgent care", "checkup", "primary care", "physician"],
    }
    for cat, words in hints.items():
        if any(w in low for w in words):
            category = cat
            break
    z = re.search(r"\b85\d{3}\b", text)
    emergency = any(x in low for x in ["can't breathe", "cannot breathe", "chest pain", "overdose", "suicid"])
    price_keywords = ["cost", "price", "how much", "cheapest", "affordable", "compare price", "cash price", "self-pay"]
    price_query = any(k in low for k in price_keywords)
    procedure = None
    if price_query:
        for proc in ["mri", "colonoscopy", "x-ray", "ct scan", "knee replacement", "knee arthroplasty"]:
            if proc in low:
                procedure = proc
                break
    return CareIntent(
        category=category,
        zip_code=z.group(0) if z else None,
        emergency=emergency,
        price_query=price_query,
        procedure=procedure,
    )


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


def select_best_provider(question: str, providers: list[dict], intent: CareIntent | None = None) -> dict | None:
    """Ranks candidate providers against the user's question and intent to return the single best match."""
    if not providers:
        return None

    q_lower = question.lower()
    stop_words = {
        "need", "near", "find", "looking", "cost", "much", "care", "where", "what",
        "please", "help", "want", "some", "good", "best", "give", "show", "tell",
        "with", "from", "that", "this", "have", "there", "they"
    }
    keywords = set(re.findall(r"\b[a-zA-Z]{3,}\b", q_lower)) - stop_words

    scored = []
    for p in providers:
        score = 0
        p_name = p.get("name", "").lower()
        p_spec = p.get("specialty", "").lower()
        p_cat = p.get("category", "")
        p_zip = str(p.get("zip", ""))

        # Category alignment
        if intent and intent.category != "All" and p_cat.lower() == intent.category.lower():
            score += 25

        # ZIP alignment
        if intent and intent.zip_code and p_zip.startswith(intent.zip_code.strip()):
            score += 35
        else:
            for z in re.findall(r"\b85\d{3}\b", q_lower):
                if z in p_zip:
                    score += 35
                    break

        # Keyword matches in name or specialty
        for kw in keywords:
            if kw in p_name:
                score += 10
            if kw in p_spec:
                score += 14

        # Affordability bonus for verified programs
        aff = p.get("affordability", "").lower()
        if "verified" in aff or "sliding" in aff or "charity" in aff:
            score += 8

        scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1] if scored else providers[0]

