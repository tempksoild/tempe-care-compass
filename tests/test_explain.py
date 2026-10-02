# Run: uv run python -m unittest discover -s tests -t . -v
# No network: the Cortex Messages client and Ollama chat are faked.
import json
import unittest
from datetime import date
from unittest import mock

from agent import CareAgent, CortexAgent
from llm.cortex_messages import CortexMessagesClient, CortexMessagesError
from rag import explain as rx
from rag.citations import provider_source_id
from rag.intent import parse_intent
from rag.rank import rank_providers
from rag.retrieve import Candidate
from rag.select import no_eligible_message

TODAY = date(2026, 10, 2)
Q = "therapist near 85281"
INTENT = parse_intent(Q)


def _cand(npi, name, zip_="85281", specialty="Mental Health", phone="4805551234"):
    r = {"npi": npi, "name": name, "category": "Behavioral health", "specialty": specialty,
         "address": "1 Main St", "city": "Tempe", "state": "AZ", "zip": zip_, "phone": phone,
         "last_updated": "2025-06-01", "source": "CMS NPPES via Affine"}
    return Candidate(source_id=provider_source_id(r), record=r)


TOP3 = rank_providers([_cand("1000000001", "Alpha Care"),
                       _cand("1000000002", "Beta Care", phone=""),
                       _cand("1000000003", "Gamma Care", zip_="85283")], INTENT, TODAY)


def model_json(providers, swap=False):
    items = []
    for label, p in rx.labelled(providers):
        items.append({"rank": label, "source_id": p.source_id, "heading": f"{label} — x", "reason": "r",
                      "proximity_explanation": "p", "financial_explanation": "f", "service_explanation": "s",
                      "important_unknowns": [], "comparison_to_next": "" if label == "C" else "gap",
                      "citation_source_ids": [p.source_id]})
    if swap:
        items[0]["source_id"], items[1]["source_id"] = items[1]["source_id"], items[0]["source_id"]
    return json.dumps({"query_interpretation": "q", "providers": items, "ranking_disclaimer": "d"})


class FakeMessages:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, []

    def create(self, system, user, schema=None, max_tokens=2048):
        self.calls.append({"system": system, "user": json.loads(user), "schema": schema})
        if self.error:
            raise self.error
        return self.reply


class TestZeroResult(unittest.TestCase):
    def test_names_main_constraint(self):
        msg = no_eligible_message({"wrong_category": 30, "duplicate": 2}, considered=32)
        self.assertIn("30 of 32 by different care category than requested", msg)
        self.assertIn("main constraint was: different care category than requested", msg)

    def test_nothing_retrieved(self):
        msg = no_eligible_message({}, considered=0, retrieval_note="No provider text matched the request terms.")
        self.assertIn("No provider text matched", msg)

    def test_template_carries_message(self):
        exp = rx.template_explanation(Q, INTENT, [], empty_message="none left")
        self.assertEqual(exp.providers, [])
        self.assertEqual(exp.ranking_disclaimer, "none left")


class TestPreparedRecords(unittest.TestCase):
    def test_only_selected_and_gaps(self):
        recs = rx.prepare_records(TOP3)
        self.assertEqual([r["rank"] for r in recs], ["A", "B", "C"])
        self.assertEqual([r["source_id"] for r in recs], [p.source_id for p in TOP3])
        self.assertIsNone(recs[-1]["score_gap_to_next"])
        self.assertGreater(recs[0]["score_gap_to_next"]["total"], 0)
        for r in recs:  # internal user-location IDs never offered as citations
            self.assertFalse(any(s.startswith("USER_LOCATION:") for s in r["allowed_source_ids"]))

    def test_refuses_more_than_three(self):
        with self.assertRaises(ValueError):
            rx.prepare_records(TOP3 + TOP3[:1])


class TestTemplate(unittest.TestCase):
    def test_structure_and_wording(self):
        exp = rx.template_explanation(Q, INTENT, TOP3)
        self.assertEqual([p.rank for p in exp.providers], ["A", "B", "C"])
        self.assertIn("highest-ranked match in the current dataset", exp.providers[0].reason)
        self.assertIsNone(exp.providers[-1].comparison_to_next)
        self.assertIn("ahead on", exp.providers[0].comparison_to_next)
        for e, p in zip(exp.providers, TOP3):  # miles only when a distance was calculated
            self.assertEqual("miles" in e.proximity_explanation, p.scores.distance_miles is not None)
        text = exp.model_dump_json().lower()
        for banned in ("best provider", "accepting new patients", "walk-in available", "free care"):
            self.assertNotIn(banned, text)


class TestMessagesClient(unittest.TestCase):
    def test_request_shape_pat(self):
        c = CortexMessagesClient("abc-xyz.snowflakecomputing.com", "PAT123", model="claude-sonnet-4-5")
        resp = mock.Mock(status_code=200, json=lambda: {"content": [{"type": "text", "text": "{}"}]})
        with mock.patch("llm.cortex_messages.requests.post", return_value=resp) as post:
            self.assertEqual(c.create("sys", "usr", {"type": "object"}), "{}")
        url, kw = post.call_args.args[0], post.call_args.kwargs
        self.assertEqual(url, "https://abc-xyz.snowflakecomputing.com/api/v2/cortex/v1/messages")
        self.assertEqual(kw["headers"]["Authorization"], "Bearer PAT123")
        self.assertEqual(kw["headers"]["anthropic-version"], "2023-06-01")
        self.assertEqual(kw["json"]["output_config"]["format"]["type"], "json_schema")
        self.assertEqual(kw["json"]["system"], "sys")

    def test_from_connection_uses_session_token(self):
        con = mock.Mock(host="h.snowflakecomputing.com")
        con.rest.token = "SESS"
        c = CortexMessagesClient.from_connection(con)
        self.assertEqual(c._headers()["Authorization"], 'Snowflake Token="SESS"')

    def test_http_error(self):
        c = CortexMessagesClient("h", "t")
        with mock.patch("llm.cortex_messages.requests.post", return_value=mock.Mock(status_code=403, text="no")):
            with self.assertRaises(CortexMessagesError):
                c.create("s", "u")


class TestCortexAgent(unittest.TestCase):
    def test_llm_path(self):
        fake = FakeMessages(reply=model_json(TOP3))
        agent = CortexAgent(connection=None, messages_client=fake)
        exp = agent.explain_ranking(Q, INTENT, TOP3)
        self.assertEqual(agent.last_explain_source, "llm")
        self.assertEqual([p.source_id for p in exp.providers], [p.source_id for p in TOP3])
        self.assertIsNone(exp.providers[-1].comparison_to_next)
        self.assertEqual([p.heading for p in exp.providers],
                         ["A — Best overall match", "B — Strong alternative", "C — Third option"])
        call = fake.calls[0]
        self.assertIn("Do not select, reorder, remove or add providers", call["system"])
        self.assertEqual(call["schema"], rx.EXPLANATION_SCHEMA)
        self.assertEqual(len(call["user"]["ranked_providers"]), 3)  # only the selected three

    def test_reordered_output_falls_back(self):
        agent = CortexAgent(connection=None, messages_client=FakeMessages(reply=model_json(TOP3, swap=True)))
        exp = agent.explain_ranking(Q, INTENT, TOP3)
        self.assertEqual(agent.last_explain_source, "template")
        self.assertIn("changed the ranking", agent.last_error)
        self.assertEqual([p.source_id for p in exp.providers], [p.source_id for p in TOP3])

    def test_api_error_falls_back(self):
        agent = CortexAgent(connection=None, messages_client=FakeMessages(error=CortexMessagesError("429")))
        exp = agent.explain_ranking(Q, INTENT, TOP3)
        self.assertEqual(agent.last_explain_source, "template")
        self.assertEqual(len(exp.providers), 3)

    def test_zero_providers_skips_llm(self):
        fake = FakeMessages(reply=model_json(TOP3))
        exp = CortexAgent(None, fake).explain_ranking(Q, INTENT, [], empty_message="none")
        self.assertEqual(fake.calls, [])
        self.assertEqual(exp.ranking_disclaimer, "none")

    def test_existing_methods_untouched(self):
        agent = CortexAgent(connection=None, messages_client=FakeMessages())
        self.assertEqual(agent.parse("dentist near 85281").category, "Dental")


class TestOllamaAgent(unittest.TestCase):
    def test_ollama_uses_schema_and_falls_back(self):
        agent = CareAgent()
        with mock.patch.object(CareAgent, "chat", return_value=model_json(TOP3)) as chat:
            exp = agent.explain_ranking(Q, INTENT, TOP3)
        self.assertEqual(agent.last_explain_source, "llm")
        self.assertEqual(chat.call_args.args[1], rx.EXPLANATION_SCHEMA)
        self.assertEqual(len(exp.providers), 3)
        with mock.patch.object(CareAgent, "chat", side_effect=ConnectionError("ollama down")):
            agent.explain_ranking(Q, INTENT, TOP3)
        self.assertEqual(agent.last_explain_source, "template")


if __name__ == "__main__":
    unittest.main()
