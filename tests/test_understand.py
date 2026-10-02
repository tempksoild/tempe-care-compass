# Run: uv run python -m unittest discover -s tests -t . -v
import json
import unittest
from datetime import date

from agent import CortexAgent
from rag.intent import parse_intent
from rag.service import ProviderRecommendationService
from rag.understand import PLAN_SCHEMA, QueryPlan, fallback_plan, merge_plan, needs_understanding, parse_plan
from repository import DemoRepository

TODAY = date(2026, 10, 2)


class TestTrigger(unittest.TestCase):
    def test_triggers(self):
        for q in ["I have a fever", "I think I broke my arm", "feeling sick", "my address is 123 Mill Ave",
                  "what's near my location", "clinic close to ASU", "sprained ankle"]:
            self.assertTrue(needs_understanding(q), q)

    def test_plain_requests_skip_llm(self):
        for q in ["dentist 85282", "pharmacy", "therapist for anxiety"]:
            self.assertFalse(needs_understanding(q), q)


class TestFallbackPlan(unittest.TestCase):
    def test_broken_arm(self):
        p = fallback_plan("I broke my arm, I'm at 700 S Mill Ave")
        self.assertEqual(p.urgency, "urgent")
        self.assertIn("urgent care", p.search_terms)
        self.assertIn("orthopedic", p.search_terms)
        self.assertEqual(p.location_text, "700 S Mill Ave")

    def test_fever(self):
        p = fallback_plan("my kid has a fever near 85283")
        self.assertIn("urgent care", p.search_terms)
        self.assertIn("pediatrics", p.search_terms)
        self.assertEqual(p.zip_code, "85283")

    def test_landmark(self):
        self.assertEqual(fallback_plan("sick, near ASU").location_text.lower(), "asu")


class TestPlanParsing(unittest.TestCase):
    def test_sanitizes_llm_output(self):
        raw = json.dumps({"care_type": "urgent care", "category": "Made Up", "search_terms":
                          ["Urgent Care", "urgent care", "x", "<script>", "orthopedic", "a" * 60],
                          "urgency": "urgent", "location_text": "700 S Mill Ave", "zip_code": "zip 85281!",
                          "reason": "r"})
        p = parse_plan(raw)
        self.assertEqual(p.category, "All")
        self.assertEqual(p.search_terms, ["urgent care", "script", "orthopedic"])
        self.assertEqual(p.zip_code, "85281")
        self.assertEqual(p.source, "llm")
        self.assertEqual(set(PLAN_SCHEMA["required"]), set(PLAN_SCHEMA["properties"]))

    def test_merge(self):
        intent = parse_intent("I broke my arm")
        m = merge_plan(intent, QueryPlan(category="Clinic / primary care", search_terms=["urgent care", "orthopedic"],
                                         zip_code="85281"), symptom_request=True)
        self.assertEqual(m.category, "All")          # orthopedics lives in "Other care"
        self.assertIn("Dental", m.excluded_categories)
        self.assertEqual(m.keywords, ["urgent care", "orthopedic"])  # user's symptom words dropped
        self.assertTrue(m.term_priority)
        self.assertEqual(m.zip_code, "85281")

    def test_llm_emergency_flag(self):
        m = merge_plan(parse_intent("my arm bone is sticking out"), QueryPlan(urgency="emergency"), True)
        self.assertTrue(m.emergency)

    def test_dentist_kept_strict(self):
        m = merge_plan(parse_intent("toothache, need a dentist"), QueryPlan(category="All"), True)
        self.assertEqual(m.category, "Dental")


class FakeMessages:
    def __init__(self, plan):
        self.plan, self.calls = plan, []

    def create(self, system, user, schema=None, max_tokens=2048):
        self.calls.append((system, user, schema))
        if schema is PLAN_SCHEMA:
            return json.dumps(self.plan)
        raise RuntimeError("explanation not needed in this test")  # -> template explanation


def fake_geocoder(address, city, state, zip_code):
    return {"latitude": 33.4148, "longitude": -111.9093, "source": "Fake Geocoder", "formatted_address": address}


def fallback_geocoder(address, city, state, zip_code):
    return {"latitude": 33.42, "longitude": -111.93, "source": "Tempe Postal Centroid", "formatted_address": address}


class TestServiceWithUnderstanding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.svc = ProviderRecommendationService(DemoRepository())

    def test_llm_reads_whole_request_then_rag(self):
        plan = {"care_type": "urgent care or orthopedics", "category": "All",
                "search_terms": ["urgent care", "orthopedic", "sports medicine"], "urgency": "urgent",
                "location_text": "700 S Mill Ave", "zip_code": "", "reason": "Possible fractures are seen at urgent care."}
        fake = FakeMessages(plan)
        agent = CortexAgent(None, fake)
        res = self.svc.recommend("I think I broke my arm skateboarding, I'm at 700 S Mill Ave",
                                 agent=agent, geocoder=fake_geocoder, today=TODAY)
        self.assertEqual(fake.calls[0][1], "I think I broke my arm skateboarding, I'm at 700 S Mill Ave")  # whole request
        self.assertEqual(res.understanding_source, "llm")
        self.assertEqual(len(res.providers), 3)
        self.assertEqual(res.providers[0].specialty, "Urgent Care")  # first plan term wins
        for p in res.providers:
            self.assertNotIn(p.category, {"Dental", "Pharmacy", "Behavioral health"})
            self.assertRegex(p.specialty.lower(), "urgent|orthop|sports")
            self.assertIsNotNone(p.scores.distance_miles)  # measured from the geocoded address
        self.assertIn("700 S Mill Ave", res.location_used)
        self.assertEqual(res.user_coordinates, (33.4148, -111.9093))
        self.assertIn("urgent care or orthopedics", res.care_guidance)
        self.assertTrue(any("same-day care" in n for n in res.limitations))

    def test_fallback_without_llm(self):
        res = self.svc.recommend("I have a fever and chills near 85281", today=TODAY)
        self.assertEqual(res.understanding_source, "fallback")
        self.assertTrue(res.providers)
        self.assertTrue(any("urgent care" in p.specialty.lower() or "family" in p.specialty.lower()
                            for p in res.providers))

    def test_geocoder_centroid_fallback_not_used_as_point(self):
        res = self.svc.recommend("sick, I live at 1 E Main St 85281", geocoder=fallback_geocoder, today=TODAY)
        self.assertIsNone(res.user_coordinates)
        self.assertIn("Could not pinpoint", res.location_used)

    def test_llm_emergency_short_circuits(self):
        agent = CortexAgent(None, FakeMessages({"care_type": "", "category": "All", "search_terms": [],
                                                "urgency": "emergency", "location_text": "", "zip_code": "",
                                                "reason": ""}))
        res = self.svc.recommend("my arm bone is poking through the skin", agent=agent, today=TODAY)
        self.assertTrue(res.emergency)
        self.assertEqual(res.providers, [])

    def test_regex_emergency_skips_llm(self):
        fake = FakeMessages({})
        res = self.svc.recommend("chest pain and fever", agent=CortexAgent(None, fake), today=TODAY)
        self.assertTrue(res.emergency)
        self.assertEqual(fake.calls, [])


if __name__ == "__main__":
    unittest.main()
