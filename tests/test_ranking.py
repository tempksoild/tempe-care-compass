# Run: uv run python -m unittest discover -s tests -t . -v
import unittest
from datetime import date

from rag import config as cfg
from rag.citations import provider_source_id
from rag.filter import EligibilityFilter
from rag.intent import parse_intent
from rag.rank import proximity_score, rank_providers, sort_key, weighted_score
from rag.retrieve import Candidate, ProviderRetriever, retrieve_for_question
from repository import DemoRepository

TODAY = date(2026, 10, 2)


def cand(sem=None, **kw):
    r = {"npi": "1234567890", "name": "Test Provider", "category": "Behavioral health", "specialty": "",
         "address": "1 Main St", "city": "Tempe", "state": "AZ", "zip": "85281", "phone": "4805551234",
         "last_updated": "2025-06-01", "source": "CMS NPPES via Affine"}
    r.update(kw)
    return Candidate(source_id=provider_source_id(r), record=r, semantic_score=sem)


class TestFormulae(unittest.TestCase):
    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(cfg.WEIGHTS.values()), 1.0)

    def test_weighted_score_renormalizes(self):
        self.assertAlmostEqual(weighted_score({"service": 80, "proximity": None, "financial": None,
                                               "confidence": 60, "availability": None}),
                               (0.35 * 80 + 0.10 * 60) / 0.45)
        self.assertEqual(weighted_score({"service": None}), 0.0)

    def test_proximity_bands(self):
        self.assertEqual([proximity_score(d) for d in (None, 0.5, 2, 4, 8, 15, 30)], [40, 100, 85, 70, 45, 20, 0])


class TestServiceScore(unittest.TestCase):
    def test_tiers(self):
        intent = parse_intent("therapist for anxiety")
        ranked = rank_providers([
            cand(npi="1000000001", name="A", specialty="Mental Health"),        # specialty match
            cand(npi="1000000002", name="Tempe Anxiety Care Group"),            # category + name keyword
            cand(npi="1000000003", name="Jane Smith"),                          # category only
        ], intent, TODAY)
        by = {p.npi: p.scores.service_match for p in ranked}
        self.assertEqual(by, {"1000000001": 100, "1000000002": 85, "1000000003": 70})

    def test_semantic_only_range(self):
        ranked = rank_providers([cand(category="Other care", name="Zed", sem=0.25)], parse_intent("wellness"), TODAY)
        self.assertTrue(40 <= ranked[0].scores.service_match <= 69)


class TestConfidence(unittest.TestCase):
    def test_full_and_sparse(self):
        intent = parse_intent("therapist")
        full = cand(npi="1000000001", specialty="Mental Health")
        sparse = cand(npi="1000000002", address="", phone="", last_updated="2010-01-01")
        by = {p.npi: p.scores.confidence for p in rank_providers([full, sparse], intent, TODAY)}
        self.assertEqual(by["1000000001"], 100)
        self.assertEqual(by["1000000002"], 35)  # active only


class TestOrdering(unittest.TestCase):
    def test_same_zip_beats_other_zip(self):
        intent = parse_intent("therapist near 85281")
        ranked = rank_providers([cand(npi="1000000001", name="Far", zip="85283"),
                                 cand(npi="1000000002", name="Near", zip="85281")], intent, TODAY)
        self.assertEqual([p.name for p in ranked], ["Near", "Far"])
        self.assertEqual(ranked[0].scores.distance_method, "same_zip")
        self.assertIsNone(ranked[0].scores.distance_miles)  # ZIP match is never a distance

    def test_coordinates_give_miles(self):
        intent = parse_intent("therapist near 85281")
        ranked = rank_providers([cand(latitude="33.4255", longitude="-111.9400")], intent, TODAY,
                                user_coords=(33.4255, -111.9400))
        self.assertEqual(ranked[0].scores.distance_method, "provider_coordinates")
        self.assertEqual(ranked[0].scores.distance_miles, 0.0)
        self.assertEqual(ranked[0].scores.proximity, 100)

    def test_tie_break_alphabetical_then_deterministic(self):
        intent = parse_intent("therapist")
        cs = [cand(npi=f"100000000{i}", name=n) for i, n in enumerate(["beta", "Alpha", "gamma"])]
        names = [p.name for p in rank_providers(cs, intent, TODAY)]
        self.assertEqual(names, ["Alpha", "beta", "gamma"])
        self.assertEqual(names, [p.name for p in rank_providers(list(reversed(cs)), intent, TODAY)])

    def test_tie_break_service_before_name(self):
        # Equal totals forced via sort_key inputs: higher service wins over name.
        intent = parse_intent("therapist")
        a, b = rank_providers([cand(npi="1000000001", name="Aaa"),
                               cand(npi="1000000002", name="Zzz", specialty="Mental Health")], intent, TODAY)
        a.scores.total = b.scores.total = 50.0
        self.assertEqual(sorted([a, b], key=sort_key)[0].name, "Zzz")

    def test_verified_affordability_ranks_higher_when_requested(self):
        intent = parse_intent("affordable counseling near 85281")
        ranked = rank_providers([
            cand(npi="1000000001", name="Aaa Unknown"),
            cand(npi="1000000002", name="Zzz Sliding", program_type="Sliding fee scale",
                 verification_url="https://example.org"),
        ], intent, TODAY)
        self.assertEqual(ranked[0].name, "Zzz Sliding")
        self.assertEqual(ranked[0].scores.financial, 90)
        self.assertEqual(ranked[1].scores.financial, 30)


class TestDemoData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = ProviderRetriever(DemoRepository())

    def test_end_to_end_sorted_and_deterministic(self):
        q = "therapist for anxiety near 85281"
        res = retrieve_for_question(q, self.r)
        elig = EligibilityFilter().apply(res.candidates, res.intent).eligible
        ranked = rank_providers(elig, res.intent, TODAY)
        self.assertEqual(len(ranked), len(elig))
        totals = [p.scores.total for p in ranked]
        self.assertEqual(totals, sorted(totals, reverse=True))
        again = rank_providers(elig, res.intent, TODAY)
        self.assertEqual([p.source_id for p in ranked], [p.source_id for p in again])
        for p in ranked:
            self.assertTrue(0 <= p.scores.total <= 100)
            self.assertIsNone(p.scores.availability)
            if p.scores.distance_method == "same_zip":  # a ZIP match is never a distance
                self.assertIsNone(p.scores.distance_miles)
            if p.scores.distance_miles is not None:
                self.assertEqual(p.scores.distance_method, "zip_centroid")


if __name__ == "__main__":
    unittest.main()
