# Run: uv run python -m unittest discover -s tests -v
import unittest

from rag.citations import attach_source_ids
from rag.intent import parse_intent
from rag.retrieve import ProviderRetriever, reciprocal_rank_fusion, retrieve_for_question
from repository import DemoRepository


class FakeRepo:
    """Small in-memory corpus; counts calls to prove retrieval hits the repo, not sidebar rows."""
    ROWS = [
        {"npi": "1", "name": "Calm Mind Counseling", "category": "Behavioral health", "specialty": "Mental Health", "zip": "85281"},
        {"npi": "2", "name": "Jane Doe Lpc", "category": "Behavioral health", "specialty": "Clinical", "zip": "85282"},
        {"npi": "3", "name": "Tempe Family Clinic", "category": "Clinic / primary care", "specialty": "Family", "zip": "85281"},
        {"npi": "4", "name": "Desert Pharmacy", "category": "Pharmacy", "specialty": "Community/Retail Pharmacy", "zip": "85283"},
        {"npi": "5", "name": "Bright Smiles", "category": "Dental", "specialty": "General Practice", "zip": "85281"},
    ]

    def __init__(self):
        self.calls = 0

    def all_providers(self):
        self.calls += 1
        return attach_source_ids([dict(r, address="", city="Tempe") for r in self.ROWS])


class TestRRF(unittest.TestCase):
    def test_fusion_rewards_agreement_and_is_deterministic(self):
        fused = reciprocal_rank_fusion(["a", "b", "c"], ["b", "d"])
        self.assertEqual(fused[0][0], "b")  # in both lists
        self.assertEqual([s for s, _ in fused], [s for s, _ in reciprocal_rank_fusion(["a", "b", "c"], ["b", "d"])])
        self.assertEqual(set(s for s, _ in fused), {"a", "b", "c", "d"})


class TestRetrieverFake(unittest.TestCase):
    def setUp(self):
        self.repo = FakeRepo()
        self.r = ProviderRetriever(self.repo)

    def test_fresh_repo_query(self):
        res = retrieve_for_question("therapist near 85281", self.r)
        self.assertEqual(self.repo.calls, 1)
        self.assertGreater(len(res.candidates), 0)

    def test_semantic_finds_counselor_for_therapist(self):
        # No record contains the word "therapist"; synonyms + trigrams must still find behavioral health.
        res = retrieve_for_question("I need a therapist", self.r)
        top = res.candidates[0].record
        self.assertEqual(top["category"], "Behavioral health")
        self.assertIsNotNone(res.candidates[0].semantic_rank)

    def test_emergency_short_circuits_without_repo_call(self):
        res = retrieve_for_question("chest pain, need a hospital", self.r)
        self.assertTrue(res.short_circuited)
        self.assertEqual(res.candidates, [])
        self.assertEqual(self.repo.calls, 0)

    def test_retrieval_does_not_hard_filter(self):
        # Wrong-category removal belongs to eligibility filtering (task 4), not retrieval.
        res = retrieve_for_question("clinic near 85281", self.r)
        cats = {c.record["category"] for c in res.candidates}
        self.assertIn("Clinic / primary care", cats)
        self.assertGreater(len(cats), 1)  # zip match pulls other categories in too

    def test_no_terms_returns_empty(self):
        res = retrieve_for_question("please help", self.r)
        self.assertEqual(res.candidates, [])
        self.assertIsNotNone(res.note)


class TestRetrieverDemoData(unittest.TestCase):
    """Runs against data/tempe_nppes_demo.csv (6,183 rows)."""

    @classmethod
    def setUpClass(cls):
        cls.r = ProviderRetriever(DemoRepository())

    def test_pool_size_and_unique_ids(self):
        res = retrieve_for_question("therapist for anxiety near 85281", self.r)
        self.assertEqual(res.corpus_size, 6183)
        self.assertLessEqual(len(res.candidates), 40)
        self.assertGreaterEqual(len(res.candidates), 30)
        self.assertLessEqual(res.lexical_hits, 50)
        self.assertLessEqual(res.semantic_hits, 50)
        ids = [c.source_id for c in res.candidates]
        self.assertEqual(len(ids), len(set(ids)))

    def test_behavioral_dominates_therapist_query(self):
        res = retrieve_for_question("therapist for anxiety near 85281", self.r)
        top10 = [c.record["category"] for c in res.candidates[:10]]
        self.assertEqual(top10[0], "Behavioral health")
        # Physical therapists ("... Physical Therapist") legitimately match "therapist"
        # lexically; the category filter in task 4 removes them, retrieval must not.
        self.assertGreaterEqual(top10.count("Behavioral health"), 6, top10)

    def test_pharmacy_query(self):
        res = retrieve_for_question("pharmacy for my prescription", self.r)
        top10 = [c.record["category"] for c in res.candidates[:10]]
        self.assertGreaterEqual(top10.count("Pharmacy"), 8, top10)

    def test_deterministic(self):
        q = "dentist near 85282"
        a = [c.source_id for c in retrieve_for_question(q, self.r).candidates]
        b = [c.source_id for c in retrieve_for_question(q, self.r).candidates]
        self.assertEqual(a, b)

    def test_intent_drives_query_not_sidebar(self):
        # Different questions produce different pools from the same corpus.
        a = {c.source_id for c in retrieve_for_question("dentist", self.r).candidates}
        b = {c.source_id for c in retrieve_for_question("pharmacy", self.r).candidates}
        self.assertFalse(a & b)


if __name__ == "__main__":
    unittest.main()
