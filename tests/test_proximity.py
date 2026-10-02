# Run: uv run python -m unittest discover -s tests -t . -v
import unittest
from datetime import date

from rag.citations import provider_source_id
from rag.geo import haversine_miles, user_location, zip_centroid, zip_centroids
from rag.intent import parse_intent
from rag.rank import proximity_fact, rank_providers
from rag.retrieve import Candidate
from rag.schemas import EvidenceState

TODAY = date(2026, 10, 2)


def cand(**kw):
    r = {"npi": "1234567890", "name": "P", "category": "Behavioral health", "specialty": "Mental Health",
         "address": "1 Main St", "city": "Tempe", "state": "AZ", "zip": "85281", "phone": "4805551234",
         "last_updated": "2025-06-01", "source": "CMS NPPES via Affine", **kw}
    return Candidate(source_id=provider_source_id(r), record=r)


class TestGeo(unittest.TestCase):
    def test_centroids_loaded(self):
        self.assertGreater(len(zip_centroids()), 100)
        lat, lon = zip_centroid("85281")
        self.assertAlmostEqual(lat, 33.43, places=1)
        self.assertAlmostEqual(lon, -111.93, places=1)
        self.assertIsNone(zip_centroid("85287"))  # ASU PO-box ZIP has no ZCTA

    def test_haversine(self):
        self.assertAlmostEqual(haversine_miles((33.0, -112.0), (34.0, -112.0)), 69.1, places=0)

    def test_user_location_methods(self):
        self.assertEqual(user_location(None, (1.0, 2.0))[1], "user_coordinates")
        self.assertEqual(user_location("85281", None)[1], "zip_centroid")
        self.assertEqual(user_location("85287", None), (None, "zip_only"))
        self.assertEqual(user_location(None, None), (None, None))


class TestFallbackChain(unittest.TestCase):
    def _fact(self, q, user_coords=None, **kw):
        return proximity_fact(cand(**kw), parse_intent(q), user_coords)

    def test_exact_coordinates(self):
        f, d, m = self._fact("therapist", user_coords=(33.4274, -111.9340), latitude="33.4274", longitude="-111.9340")
        self.assertEqual((m, d, f.score), ("provider_coordinates", 0.0, 100))

    def test_zip_centroid_distance(self):
        f, d, m = self._fact("therapist near 85281", zip="85283")
        self.assertEqual(m, "zip_centroid")
        self.assertAlmostEqual(d, 4.3, delta=0.3)
        self.assertIn("ZIP centers", f.explanation)
        self.assertIn("GEO:zcta_centroid:85283", f.source_ids)

    def test_same_zip_never_miles(self):
        f, d, m = self._fact("therapist near 85281")
        self.assertEqual((m, d), ("same_zip", None))
        self.assertNotIn("mile", f.explanation)

    def test_same_city_when_no_centroid(self):
        f, d, m = self._fact("therapist near 85281", zip="85287")
        self.assertEqual((m, d, f.score), ("same_city", None, 50))

    def test_unknown(self):
        f, d, m = self._fact("therapist near 85281", zip="", city="Mesa")
        self.assertEqual(m, "unknown")
        self.assertEqual(f.evidence_state, EvidenceState.UNKNOWN)

    def test_no_user_location(self):
        f, d, m = self._fact("therapist")
        self.assertIsNone(f.score)
        self.assertEqual(f.evidence_state, EvidenceState.NOT_APPLICABLE)


class TestProximityOrdering(unittest.TestCase):
    def test_same_zip_then_adjacent_then_far(self):
        intent = parse_intent("therapist near 85281")
        ranked = rank_providers([cand(npi="1000000001", name="Far", zip="85284"),
                                 cand(npi="1000000002", name="Adjacent", zip="85282"),
                                 cand(npi="1000000003", name="Same", zip="85281")], intent, TODAY)
        self.assertEqual([p.name for p in ranked], ["Same", "Adjacent", "Far"])
        self.assertEqual([p.scores.distance_method for p in ranked], ["same_zip", "zip_centroid", "zip_centroid"])


if __name__ == "__main__":
    unittest.main()
