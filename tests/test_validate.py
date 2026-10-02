# Run: uv run python -m unittest discover -s tests -t . -v
import json
import unittest
from datetime import date

from agent import CortexAgent
from rag import explain as rx
from rag.citations import provider_source_id
from rag.intent import parse_intent
from rag.rank import rank_providers
from rag.retrieve import Candidate
from rag.validate import ExplanationValidationError, validate, validate_explanation

TODAY = date(2026, 10, 2)
Q = "therapist near 85281"
INTENT = parse_intent(Q)


def _cand(npi, name, zip_="85281", **kw):
    r = {"npi": npi, "name": name, "category": "Behavioral health", "specialty": "Mental Health",
         "address": "1 Main St", "city": "Tempe", "state": "AZ", "zip": zip_, "phone": "4805551234",
         "last_updated": "2025-06-01", "source": "CMS NPPES via Affine", **kw}
    return Candidate(source_id=provider_source_id(r), record=r)


TOP3 = rank_providers([_cand("1000000001", "Alpha"), _cand("1000000002", "Beta", phone=""),
                       _cand("1000000003", "Gamma", zip_="85283")], INTENT, TODAY)


def good():
    return rx.template_explanation(Q, INTENT, TOP3)


def edit(fn):
    exp = good()
    fn(exp)
    return exp


class TestStructure(unittest.TestCase):
    def test_template_is_valid(self):
        self.assertEqual(validate(good(), TOP3), [])

    def test_template_valid_for_two_and_one(self):
        for n in (1, 2):
            self.assertEqual(validate(rx.template_explanation(Q, INTENT, TOP3[:n]), TOP3[:n]), [])

    def test_reordered(self):
        exp = edit(lambda e: e.providers.reverse())
        self.assertTrue(any("A,B,C prefix" in v for v in validate(exp, TOP3)))

    def test_swapped_ids(self):
        def swap(e):
            e.providers[0].source_id, e.providers[1].source_id = e.providers[1].source_id, e.providers[0].source_id
        self.assertIn("providers were reordered", validate(edit(swap), TOP3))

    def test_replaced_provider(self):
        exp = edit(lambda e: setattr(e.providers[2], "source_id", "NPPES:9999999999:address:deadbeef"))
        self.assertIn("source-ID set differs from the selected providers", validate(exp, TOP3))

    def test_dropped_provider(self):
        exp = edit(lambda e: e.providers.pop())
        self.assertTrue(validate(exp, TOP3))

    def test_citation_outside_supplied(self):
        exp = edit(lambda e: e.providers[0].citation_source_ids.append("HRSA:made-up"))
        self.assertTrue(any("citations not in supplied IDs" in v for v in validate(exp, TOP3)))

    def test_user_location_not_citable(self):
        exp = edit(lambda e: e.providers[0].citation_source_ids.append("USER_LOCATION:zip_centroid:request"))
        self.assertTrue(validate(exp, TOP3))

    def test_raises(self):
        with self.assertRaises(ExplanationValidationError):
            validate_explanation(edit(lambda e: e.providers.reverse()), TOP3)


class TestClaims(unittest.TestCase):
    def _bad(self, field, text, idx=0):
        exp = edit(lambda e: setattr(e.providers[idx], field, text))
        return validate(exp, TOP3)

    def test_unverified_free(self):
        self.assertTrue(self._bad("financial_explanation", "This clinic offers free care."))
        self.assertTrue(self._bad("reason", "Alpha is affordable and close."))

    def test_hedged_affordability_ok(self):
        self.assertEqual(self._bad("financial_explanation", "Whether sliding-fee care is offered is not verified."), [])

    def test_unverified_availability(self):
        self.assertTrue(self._bad("reason", "Alpha is accepting new patients"))
        self.assertTrue(self._bad("service_explanation", "Walk-ins welcome"))
        self.assertEqual(self._bad("reason", "Call to confirm whether they accept new patients."), [])

    def test_distance_from_zip_match(self):
        # A is a same-ZIP match (no distance) -> any miles figure is invented.
        self.assertIsNone(TOP3[0].scores.distance_miles)
        self.assertTrue(self._bad("proximity_explanation", "About 0.5 miles away."))

    def test_real_distance_allowed(self):
        c = TOP3[2]
        self.assertIsNotNone(c.scores.distance_miles)
        self.assertEqual(self._bad("proximity_explanation", f"About {c.scores.distance_miles} miles between ZIP centers.", 2), [])
        self.assertTrue(self._bad("proximity_explanation", "About 9.9 miles away.", 2))

    def test_expensive_without_price(self):
        self.assertTrue(self._bad("financial_explanation", "Likely more expensive than B."))

    def test_quality_claim(self):
        self.assertTrue(self._bad("reason", "Alpha is the best provider in Tempe."))
        exp = edit(lambda e: setattr(e, "ranking_disclaimer", "These are top-rated doctors."))
        self.assertTrue(validate(exp, TOP3))

    def test_verified_affordability_may_be_stated(self):
        p = rank_providers([_cand("1000000009", "Sliding Co", program_type="Sliding fee scale",
                                  verification_url="https://example.org")], parse_intent("affordable therapist"), TODAY)
        exp = rx.template_explanation("affordable therapist", parse_intent("affordable therapist"), p)
        exp.providers[0].financial_explanation = "Offers a verified sliding-fee program."
        self.assertEqual(validate(exp, p), [])


class TestAgentFallback(unittest.TestCase):
    class Fake:
        def __init__(self, reply):
            self.reply = reply

        def create(self, *a, **k):
            return self.reply

    def _reply(self, mutate):
        data = json.loads(good().model_dump_json())
        for item in data["providers"]:
            item["comparison_to_next"] = item["comparison_to_next"] or ""
        mutate(data)
        return json.dumps(data)

    def test_claim_violation_falls_back_to_template(self):
        reply = self._reply(lambda d: d["providers"][0].update(reason="Alpha offers free walk-in care."))
        agent = CortexAgent(None, self.Fake(reply))
        exp = agent.explain_ranking(Q, INTENT, TOP3)
        self.assertEqual(agent.last_explain_source, "template")
        self.assertIn("unverified", agent.last_error)
        self.assertEqual(validate(exp, TOP3), [])

    def test_clean_reply_accepted(self):
        agent = CortexAgent(None, self.Fake(self._reply(lambda d: None)))
        agent.explain_ranking(Q, INTENT, TOP3)
        self.assertEqual(agent.last_explain_source, "llm")


if __name__ == "__main__":
    unittest.main()
