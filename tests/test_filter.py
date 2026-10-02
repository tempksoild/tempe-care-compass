# Run: uv run python -m unittest discover -s tests -t . -v
import unittest

from rag.citations import provider_source_id
from rag.filter import EligibilityFilter
from rag.intent import parse_intent
from rag.retrieve import Candidate, ProviderRetriever, retrieve_for_question
from repository import DemoRepository


def cand(**kw):
    r = {"npi": "1234567890", "name": "Test Provider", "category": "Behavioral health", "specialty": "",
         "address": "1 Main St", "city": "Tempe", "state": "AZ", "zip": "85281", "phone": "4805551234",
         "last_updated": "2025-01-01", "source": "CMS NPPES via Affine"}
    r.update(kw)
    return Candidate(source_id=provider_source_id(r), record=r)


class TestEligibility(unittest.TestCase):
    def setUp(self):
        self.f = EligibilityFilter()
        self.therapist = parse_intent("therapist near 85281")

    def test_valid_passes(self):
        res = self.f.apply([cand()], self.therapist)
        self.assertEqual(len(res.eligible), 1)
        self.assertEqual(res.eliminated_by, {})

    def test_each_constraint_counted(self):
        cands = [
            cand(npi="12"),                                   # invalid NPI
            cand(npi="1111111111", name=""),                  # invalid name
            cand(npi="2222222222", active="false"),           # inactive
            cand(npi="3333333333", city="Phoenix"),           # outside area
            cand(npi="4444444444", category="Pharmacy"),      # wrong category
            cand(npi="5555555555"),
            cand(npi="5555555555", address="1 MAIN ST."),     # duplicate (same normalized location)
            cand(npi="5555555555", address="9 Other Rd"),     # same NPI, different location -> kept
        ]
        res = self.f.apply(cands, self.therapist)
        self.assertEqual(res.eliminated_by, {"invalid_record": 2, "inactive": 1, "outside_area": 1,
                                             "wrong_category": 1, "duplicate": 1})
        self.assertEqual(len(res.eligible), 2)
        self.assertEqual(res.considered, 8)

    def test_zip_is_ranking_factor_unless_required(self):
        c = cand(zip="85283")
        self.assertEqual(len(EligibilityFilter().apply([c], self.therapist).eligible), 1)
        self.assertEqual(EligibilityFilter(require_zip=True).apply([c], self.therapist).eliminated_by,
                         {"outside_area": 1})

    def test_unknown_affordability_not_filtered_by_default(self):
        res = self.f.apply([cand()], parse_intent("cheap therapist, uninsured"))
        self.assertEqual(len(res.eligible), 1)

    def test_verified_only_hard_filter(self):
        intent = parse_intent("only verified sliding-fee counseling")
        verified = cand(npi="6666666666", program_type="Sliding fee scale", verification_url="https://example.org")
        unverified_claim = cand(npi="7777777777", program_type="Sliding fee scale")  # no verification source
        res = self.f.apply([cand(), verified, unverified_claim], intent)
        self.assertEqual([c.record["npi"] for c in res.eligible], ["6666666666"])
        self.assertEqual(res.eliminated_by, {"not_verified_affordable": 2})

    def test_procedure_filter_needs_price_linkage(self):
        intent = parse_intent("how much does a colonoscopy cost")
        c = cand(category="Clinic / primary care")
        unlinked = self.f.apply([c], intent)
        self.assertEqual(len(unlinked.eligible), 1)
        self.assertTrue(unlinked.limitations)
        linked = EligibilityFilter(price_index={}).apply([c], intent)
        self.assertEqual(linked.eliminated_by, {"missing_procedure": 1})
        ok = EligibilityFilter(price_index={c.source_id: {("CPT", "45378")}}).apply([c], intent)
        self.assertEqual(len(ok.eligible), 1)

    def test_describe(self):
        lines = EligibilityFilter.describe({"wrong_category": 3, "duplicate": 1})
        self.assertEqual(lines[0], "3 removed: different care category than requested")


class TestEligibilityDemoData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = ProviderRetriever(DemoRepository())

    def test_therapist_drops_physical_therapists(self):
        res = retrieve_for_question("therapist for anxiety near 85281", self.r)
        out = EligibilityFilter().apply(res.candidates, res.intent)
        self.assertTrue(out.eligible)
        self.assertEqual({c.record["category"] for c in out.eligible}, {"Behavioral health"})
        self.assertGreater(out.eliminated_by.get("wrong_category", 0), 0)
        self.assertEqual(len(out.eligible) + sum(out.eliminated_by.values()), len(res.candidates))


if __name__ == "__main__":
    unittest.main()
