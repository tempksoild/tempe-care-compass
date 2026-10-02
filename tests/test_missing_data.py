# Run: uv run python -m unittest discover -s tests -t . -v
# Unknown must never be scored or labelled as "verified no".
import unittest
from datetime import date

from rag.citations import provider_source_id
from rag.evidence import affordability_evidence, availability_evidence
from rag.intent import parse_intent
from rag.rank import rank_providers
from rag.retrieve import Candidate
from rag.schemas import EvidenceState

TODAY = date(2026, 10, 2)
BASE = {"npi": "1234567890", "name": "P", "category": "Clinic / primary care", "specialty": "Family",
        "address": "1 Main St", "city": "Tempe", "state": "AZ", "zip": "85281", "phone": "4805551234",
        "last_updated": "2025-06-01", "source": "CMS NPPES via Affine",
        "affordability": "Not stated in NPPES — call to verify"}


def one(intent_text, **kw):
    r = {**BASE, **kw}
    return rank_providers([Candidate(source_id=provider_source_id(r), record=r)], parse_intent(intent_text), TODAY)[0]


class TestMissingData(unittest.TestCase):
    def test_nppes_affordability_is_unknown(self):
        self.assertEqual(affordability_evidence(BASE).state, EvidenceState.UNKNOWN)

    def test_unknown_affordability_neutral_when_requested(self):
        p = one("cheap clinic, uninsured")
        self.assertEqual(p.scores.financial, 30)
        self.assertEqual(p.scores.financial_evidence_state, EvidenceState.UNKNOWN)
        self.assertTrue(any("not assumed expensive" in u for u in p.unknown_facts))

    def test_unknown_affordability_dropped_when_not_requested(self):
        p = one("family doctor")
        self.assertIsNone(p.scores.financial)
        self.assertEqual(p.scores.financial_evidence_state, EvidenceState.NOT_APPLICABLE)

    def test_price_query_unknown_price_is_neutral_not_zero(self):
        p = one("how much does a colonoscopy cost")
        self.assertEqual(p.scores.financial, 30)

    def test_verified_no_assistance_scores_zero(self):
        p = one("cheap clinic", program_type="No assistance offered", verification_url="https://x")
        self.assertEqual(p.scores.financial, 0)
        self.assertEqual(p.scores.financial_evidence_state, EvidenceState.VERIFIED_NO)

    def test_availability_unknown_is_none_not_false(self):
        p = one("family doctor")
        self.assertIsNone(p.scores.availability)
        self.assertEqual(p.scores.availability_evidence_state, EvidenceState.UNKNOWN)
        # An unsourced flag is still unknown.
        self.assertEqual(availability_evidence({**BASE, "accepting_new_patients": "yes"})[0], EvidenceState.UNKNOWN)

    def test_verified_availability(self):
        p = one("family doctor", walk_in="yes", availability_source="clinic website")
        self.assertEqual(p.scores.availability, 100)

    def test_no_location_proximity_not_applicable(self):
        p = one("family doctor")
        self.assertIsNone(p.scores.proximity)
        self.assertIsNone(p.scores.distance_method)


if __name__ == "__main__":
    unittest.main()
