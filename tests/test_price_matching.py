# Run: uv run python -m unittest discover -s tests -t . -v
import unittest
from datetime import date

from rag.citations import provider_source_id
from rag.filter import EligibilityFilter
from rag.intent import parse_intent
from rag.prices import build_price_index, comparable_rows, hospital_price_scores, is_local_ccn
from rag.rank import rank_providers
from rag.retrieve import Candidate
from repository import DemoRepository

TODAY = date(2026, 10, 2)
CODES = [("CPT", "45378")]


def row(ccn, amount, rate_type="cash", setting="outpatient", code="45378", snap="2026-05-22", payer=""):
    return {"ccn": ccn, "billing_code": code, "billing_code_type": "CPT", "billing_code_description": "COLONOSCOPY",
            "payer_name": payer, "setting": setting, "rate_type": rate_type, "rate_amount": str(amount),
            "snapshot_date": snap}


def hospital(npi, name, ccn):
    r = {"npi": npi, "name": name, "category": "Hospital", "specialty": "General Acute Care", "address": "1 Rd",
         "city": "Tempe", "state": "AZ", "zip": "85281", "phone": "4805551234", "last_updated": "2025-06-01",
         "source": "CMS NPPES via Affine", "ccn": ccn}
    return Candidate(source_id=provider_source_id(r), record=r)


class TestMatching(unittest.TestCase):
    def test_only_matching_code_setting_type_dates(self):
        rows = [row("030001", 100), row("030002", 200),
                row("030003", 50, code="27447"),            # other code
                row("030004", 50, setting=""),              # unknown setting
                row("030005", 50, rate_type="negotiated"),  # other price type -> other group
                row("030006", 50, snap="2025-01-01"),       # stale vs latest snapshot
                row("", 50),                                # unknown hospital
                row("030001", 100)]                         # exact duplicate
        groups = comparable_rows(rows, CODES)
        cash = groups[("CPT", "45378", "outpatient", "cash")]
        self.assertEqual(sorted(m["ccn"] for m in cash), ["030001", "030002"])
        self.assertIn(("CPT", "45378", "outpatient", "negotiated"), groups)

    def test_formula_and_equal_prices(self):
        s = hospital_price_scores([row("030001", 100), row("030002", 300), row("030003", 200)], CODES)
        self.assertEqual((s["030001"].price_score, s["030002"].price_score, s["030003"].price_score), (100, 0, 50))
        eq = hospital_price_scores([row("030001", 100), row("030002", 100)], CODES)
        self.assertEqual({v.price_score for v in eq.values()}, {100})

    def test_lowest_rate_per_hospital_and_cash_preferred(self):
        rows = [row("030001", 150), row("030001", 120), row("030002", 300),
                row("030001", 10, rate_type="negotiated"), row("030002", 20, rate_type="negotiated")]
        s = hospital_price_scores(rows, CODES)
        self.assertEqual(s["030001"].group[3], "cash")
        self.assertEqual(s["030001"].price, 120)

    def test_needs_two_hospitals(self):
        self.assertEqual(hospital_price_scores([row("030001", 100)], CODES), {})

    def test_california_excluded(self):
        self.assertFalse(is_local_ccn("050441"))
        self.assertFalse(is_local_ccn(50441))   # leading zero restored
        self.assertTrue(is_local_ccn("030001"))
        s = hospital_price_scores([row("050441", 100), row("050002", 200)], CODES)
        self.assertTrue(all(v.price_score is None and not v.price_eligible for v in s.values()))

    def test_demo_price_file_is_not_price_eligible(self):
        s = hospital_price_scores(DemoRepository().all_prices(), CODES)
        self.assertTrue(s)  # comparable groups exist...
        self.assertTrue(all(not v.price_eligible and v.price_score is None for v in s.values()))  # ...but all CA


class TestFinancialScoring(unittest.TestCase):
    def setUp(self):
        self.cands = [hospital("1000000001", "Cheap Hospital", "030001"),
                      hospital("1000000002", "Pricey Hospital", "030002"),
                      hospital("1000000003", "No Price Hospital", "030003")]
        self.prices = hospital_price_scores([row("030001", 100), row("030002", 300)], CODES)

    def test_price_query_uses_price_score(self):
        intent = parse_intent("how much does a colonoscopy cost")
        ranked = rank_providers(self.cands, intent, TODAY, hospital_prices=self.prices)
        by = {p.name: p.scores for p in ranked}
        self.assertEqual(ranked[0].name, "Cheap Hospital")
        self.assertEqual((by["Cheap Hospital"].financial, by["Cheap Hospital"].financial_basis), (100, "price"))
        self.assertEqual(by["Pricey Hospital"].financial, 0)
        self.assertEqual((by["No Price Hospital"].financial, by["No Price Hospital"].financial_basis),
                         (30, "neutral_unknown"))  # unknown is neutral, not 0
        self.assertTrue(by["Cheap Hospital"].price_eligible)
        self.assertIsNone(by["Cheap Hospital"].affordability_score)

    def test_uninsured_query_uses_affordability_not_price(self):
        intent = parse_intent("cheap hospital, I'm uninsured")
        ranked = rank_providers(self.cands, intent, TODAY, hospital_prices=self.prices)
        self.assertTrue(all(p.scores.financial_basis == "neutral_unknown" for p in ranked))

    def test_california_prices_do_not_rank(self):
        cands = [hospital("1000000001", "A", "050441"), hospital("1000000002", "B", "050002")]
        ca = hospital_price_scores([row("050441", 100), row("050002", 300)], CODES)
        ranked = rank_providers(cands, parse_intent("colonoscopy price"), TODAY, hospital_prices=ca)
        self.assertEqual({p.scores.financial for p in ranked}, {30})
        self.assertIn("out-of-state", ranked[0].facts[2].explanation)

    def test_price_index_feeds_procedure_filter(self):
        rows = [row("030001", 100), row("030002", 300)]
        idx = build_price_index(self.cands, rows)
        res = EligibilityFilter(price_index=idx).apply(self.cands, parse_intent("colonoscopy cost"))
        self.assertEqual(len(res.eligible), 2)
        self.assertEqual(res.eliminated_by, {"missing_procedure": 1})


if __name__ == "__main__":
    unittest.main()
