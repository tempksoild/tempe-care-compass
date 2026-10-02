# rag/config.py — Tunable ranking/filtering constants. Starting values, not truth:
# tune against test questions, keep tests/test_ranking.py green.

WEIGHTS = {
    "service": 0.35,
    "proximity": 0.30,
    "financial": 0.20,
    "confidence": 0.10,
    "availability": 0.05,
}

# Service match (0-100)
SERVICE_SPECIALTY_MATCH = 100      # request term found in specialty
SERVICE_CATEGORY_KEYWORD = 85      # right category + request term in name
SERVICE_CATEGORY_ONLY = 70         # right category, no term match
SERVICE_SEMANTIC_MIN = 40          # semantic similarity only: 40-69
SERVICE_SEMANTIC_MAX = 69
SERVICE_SEMANTIC_FULL_AT = 0.5     # trigram cosine that maps to SERVICE_SEMANTIC_MAX
SERVICE_WEAK = 25                  # lexical/general hit only
SERVICE_NONE = 10

# Proximity without a measurable distance. Location *matches*, never miles.
# same_zip sits above the 1-3 mi band (85) because Tempe ZIP centers are ~2-3 mi
# apart, so an adjacent ZIP should not outrank the requested one.
LOCATION_MATCH_SCORES = {
    "same_zip": 90,
    "same_city": 50,
    "unknown": 40,
}
ZIP_CENTROIDS_CSV = "data/az_zip_centroids.csv"   # Census 2024 ZCTA Gazetteer internal points

# Price comparison (task 10)
# CCN digits 1-2 are the CMS state code. Arizona = "03"; the Healthparse sample is "05" (California).
LOCAL_CCN_STATE_CODES = {"03"}
PRICE_DATE_TOLERANCE_DAYS = 180     # snapshots must be within this of the group's latest
PRICE_RATE_TYPE_PREFERENCE = ["cash", "min", "negotiated", "gross", "max"]  # self-pay first
PRICE_MIN_HOSPITALS = 2             # need at least two hospitals to compare

# Affordability evidence tiers (0-100)
AFFORDABILITY_SCORES = {
    "free": 100,
    "sliding_fee": 90,
    "discounted_self_pay": 80,
    "cash_price_only": 60,
    "unknown": 30,
    "no_assistance": 0,
}
NEUTRAL_FINANCIAL = 30             # used when financial matters but evidence is unknown

# Data confidence points (sum to 100)
CONFIDENCE_POINTS = {
    "active": 35,
    "full_address": 20,
    "phone": 15,
    "specialty": 15,
    "fresh": 15,
}
FRESH_YEARS = 2

# Service area for the "outside requested area" filter
SERVICE_AREA = {"state": "AZ", "city": "tempe"}
