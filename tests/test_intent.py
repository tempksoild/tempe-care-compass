# Run: uv run python -m unittest discover -s tests -v
import unittest

from rag.intent import detect_emergency, parse_intent


class TestEmergency(unittest.TestCase):
    def test_emergency_phrases(self):
        for q in ["I have chest pain", "my dad can't breathe", "friend overdosed", "I want to kill myself"]:
            self.assertIsNotNone(detect_emergency(q), q)

    def test_non_emergency(self):
        for q in ["need a dentist near 85281", "how much is a colonoscopy", "find a provider near me"]:
            self.assertIsNone(detect_emergency(q), q)


class TestIntent(unittest.TestCase):
    def test_therapist_zip(self):
        i = parse_intent("I need a therapist for anxiety near 85281")
        self.assertEqual(i.category, "Behavioral health")
        self.assertEqual(i.zip_code, "85281")
        self.assertIn("therapist", i.keywords)
        self.assertIn("anxiety", i.keywords)
        self.assertNotIn("85281", i.keywords)
        self.assertFalse(i.price_query)

    def test_provider_word_is_not_er(self):
        # agent._fallback_parse maps "provider " -> Hospital via "er "; this must not.
        self.assertNotEqual(parse_intent("find a provider near me").category, "Hospital")

    def test_physical_therapy_not_behavioral(self):
        self.assertEqual(parse_intent("physical therapy in tempe").category, "Other care")

    def test_price_procedure_codes(self):
        i = parse_intent("How much does a colonoscopy cost?")
        self.assertTrue(i.price_query)
        self.assertEqual(i.procedure, "colonoscopy")
        self.assertEqual(i.billing_codes, [("CPT", "45378")])

    def test_procedure_without_price_question(self):
        i = parse_intent("where can I get a colonoscopy")
        self.assertFalse(i.price_query)
        self.assertIsNone(i.procedure)

    def test_affordability_is_ranking_factor_not_filter(self):
        i = parse_intent("cheap clinic, I'm uninsured")
        self.assertTrue(i.affordability_required)
        self.assertFalse(i.verified_only)

    def test_verified_only_hard_filter(self):
        for q in ["only verified sliding-fee clinics", "sliding scale clinics only"]:
            i = parse_intent(q)
            self.assertTrue(i.verified_only, q)
            self.assertTrue(i.affordability_required, q)

    def test_emergency_flag_in_intent(self):
        i = parse_intent("chest pain, which hospital?")
        self.assertTrue(i.emergency)
        self.assertEqual(i.emergency_reason, "chest pain")


if __name__ == "__main__":
    unittest.main()
