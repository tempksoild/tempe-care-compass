# export_data.py — Refresh data/*.csv demo snapshots from CARE_AI.CURATED in Snowflake
# Usage: python scripts/export_data.py [--prices-limit N]
# Credentials: SNOWFLAKE_ACCOUNT / SNOWFLAKE_USER / SNOWFLAKE_PASSWORD (+ optional ROLE/WAREHOUSE)
import argparse, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from repository import SnowflakeRepository

PROVIDERS_SQL = """
    SELECT NPI, NAME, CATEGORY, SPECIALTY, ADDRESS, CITY, STATE, ZIP,
           PHONE, LAST_UPDATED, AFFORDABILITY, SOURCE
    FROM CARE_AI.CURATED.TEMPE_PROVIDERS
    ORDER BY NAME
"""
PRICES_SQL = """
    SELECT ccn, billing_code, billing_code_type, billing_code_description,
           payer_name, setting, rate_type, rate_amount, snapshot_date
    FROM CARE_AI.CURATED.HOSPITAL_PRICES
    ORDER BY billing_code_description, rate_amount
    LIMIT %s
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices-limit", type=int, default=1000)
    args = ap.parse_args()

    repo = SnowflakeRepository()
    out = ROOT / "data"
    out.mkdir(exist_ok=True)

    providers = pd.DataFrame(repo._query(PROVIDERS_SQL))
    providers.to_csv(out / "tempe_nppes_demo.csv", index=False)
    print(f"wrote {len(providers)} rows -> data/tempe_nppes_demo.csv")

    prices = pd.DataFrame(repo._query(PRICES_SQL, (args.prices_limit,)))
    prices.to_csv(out / "tempe_prices_demo.csv", index=False)
    print(f"wrote {len(prices)} rows -> data/tempe_prices_demo.csv")


if __name__ == "__main__":
    main()
