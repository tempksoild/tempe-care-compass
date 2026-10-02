# Run: uv run python -m unittest discover -s tests -t . -v
import json
import unittest
from datetime import date

from rag.citations import attach_source_ids
from rag.service import ProviderRecommendationService
from rag.validate import validate
from repository import DemoRepository

TODAY = date(2026, 10, 2)


class TinyRepo:
    def __init__(self, rows):
        self.rows = rows

    def all_providers(self):
        return attach_source_ids([dict(r) for r in self.rows])

    def all_prices(self):
        return []


def row(npi, name, category="Dental", zip_="85281", specialty="General Practice"):
    return {"npi": npi, "name": name, "category": category, "specialty": specialty, "address": "1 Main St",
            "city": "Tempe", "state": "AZ", "zip": zip_, "phone": "4805551234", "last_updated": "2025-06-01",
            "source": "CMS NPPES via Affine"}


class TestSelection(unittest.TestCase):
    def test_exactly_three_from_many(self):
        svc = ProviderRecommendationService(DemoRepository())
        res = svc.recommend("therapist for anxiety near 85281", today=TODAY)
        self.assertEqual([p.rank for p in res.providers], ["A", "B", "C"])
        self.assertEqual([e.rank for e in res.explanations], ["A", "B", "C"])
        self.assertEqual([e.source_id for e in res.explanations], [p.source_id for p in res.providers])
        self.assertEqual(res.explanation_source, "template")
        self.assertTrue(res.ranking_basis)
        self.assertIn("highest-ranked match in the current dataset", res.ranking_disclaimer)

    def test_two_not_padded(self):
        svc = ProviderRecommendationService(TinyRepo([row("1000000001", "Smile A"), row("1000000002", "Smile B"),
                                                      row("1000000003", "Rx Shop", category="Pharmacy",
                                                          specialty="Community/Retail Pharmacy")]))
        res = svc.recommend("dentist near 85281", today=TODAY)
        self.assertEqual([p.rank for p in res.providers], ["A", "B"])
        self.assertTrue(res.limitations[0].startswith("Only 2 providers met the requirements"))
        self.assertNotIn("Rx Shop", [p.name for p in res.providers])

    def test_one(self):
        res = ProviderRecommendationService(TinyRepo([row("1000000001", "Smile A")])).recommend("dentist", today=TODAY)
        self.assertEqual([p.rank for p in res.providers], ["A"])
        self.assertTrue(res.limitations[0].startswith("Only 1 provider met"))

    def test_zero_names_constraint(self):
        svc = ProviderRecommendationService(TinyRepo([row("1000000001", "Rx Dental Supply", category="Pharmacy")]))
        res = svc.recommend("dentist", today=TODAY)
        self.assertEqual(res.providers, [])
        self.assertIn("different care category than requested", res.limitations[0])
        self.assertEqual(res.eliminated_by, {"wrong_category": 1})

    def test_emergency_short_circuit(self):
        res = ProviderRecommendationService(DemoRepository()).recommend("my friend overdosed, which hospital")
        self.assertTrue(res.emergency)
        self.assertEqual(res.providers, [])
        self.assertIn("911", res.limitations[0])


class FakeAgent:
    """Mimics CortexAgent.explain_ranking contract but returns an invented claim."""
    last_explain_source = "llm"

    def explain_ranking(self, question, intent, providers, empty_message=None):
        from rag import explain as rx
        exp = rx.template_explanation(question, intent, providers, empty_message)
        exp.providers[0].reason = "A offers free walk-in care."
        return exp


class TestServiceGuard(unittest.TestCase):
    def test_service_revalidates_agent_output(self):
        svc = ProviderRecommendationService(DemoRepository())
        res = svc.recommend("dentist near 85282", agent=FakeAgent(), today=TODAY)
        self.assertEqual(res.explanation_source, "template")
        self.assertNotIn("free walk-in", json.dumps([e.model_dump() for e in res.explanations]))

    def test_template_output_validates(self):
        svc = ProviderRecommendationService(DemoRepository())
        for q in ["dentist near 85282", "pharmacy", "cheap family doctor near 85283, uninsured",
                  "how much does a colonoscopy cost near 85281"]:
            res = svc.recommend(q, today=TODAY)
            from rag.schemas import TopThreeExplanation
            exp = TopThreeExplanation(query_interpretation="", providers=res.explanations,
                                      ranking_disclaimer=res.ranking_disclaimer)
            self.assertEqual(validate(exp, res.providers), [], q)


if __name__ == "__main__":
    unittest.main()
