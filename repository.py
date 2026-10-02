# repository.py — Data access layer for Tempe Care Compass
# Provides two interchangeable repository classes:
#   DemoRepository      — reads local CSV files with pandas (no credentials needed)
#   SnowflakeRepository — executes parameterized SQL against CARE_AI.CURATED.*
#
# Both expose the same interface:
#   search(category, zip_code, query, limit)  -> list[dict]  (provider records)
#   search_prices(procedure, limit)           -> list[dict]  (hospital pricing rows)
#   get_price_comparison()                    -> list[dict]  (aggregated price spreads)

import os, re
from pathlib import Path
import pandas as pd


class DemoRepository:
    def search(self, category="All", zip_code="", query="", limit=50):
        df = pd.read_csv(Path(__file__).parent / "data/tempe_nppes_demo.csv", dtype=str).fillna("")
        if category != "All": df = df[df.category == category]
        if zip_code: df = df[df.zip.str.startswith(zip_code.strip())]
        if query:
            hay = df[["name","specialty","address"]].agg(" ".join, axis=1)
            df = df[hay.str.contains(re.escape(query), case=False, na=False)]
        return df.head(limit).to_dict("records")

    def search_prices(self, procedure="", limit=20):
        csv_path = Path(__file__).parent / "data/tempe_prices_demo.csv"
        if not csv_path.exists():
            return []
        df = pd.read_csv(csv_path, dtype={"rate_amount": float}).fillna("")
        if procedure:
            df = df[df.billing_code_description.str.contains(re.escape(procedure), case=False, na=False)]
        return df.head(limit).to_dict("records")

    def get_price_comparison(self):
        csv_path = Path(__file__).parent / "data/tempe_prices_demo.csv"
        if not csv_path.exists():
            return []
        df = pd.read_csv(csv_path, dtype={"rate_amount": float}).fillna("")
        grouped = df.groupby("billing_code_description").agg(
            hospital_count=("ccn", "nunique"),
            min_price=("rate_amount", "min"),
            max_price=("rate_amount", "max"),
            avg_price=("rate_amount", "mean"),
        ).reset_index()
        grouped["price_spread"] = grouped["max_price"] - grouped["min_price"]
        return grouped.sort_values("price_spread", ascending=False).to_dict("records")


def _load_snowflake_secrets():
    """Read the [snowflake] table from .streamlit/secrets.toml, or {} if unavailable."""
    path = Path(__file__).parent / ".streamlit" / "secrets.toml"
    if not path.exists():
        return {}
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:
        import toml as tomllib  # bundled with streamlit on older Pythons
        return tomllib.load(path).get("snowflake", {})
    with path.open("rb") as f:
        return tomllib.load(f).get("snowflake", {})


class SnowflakeRepository:
    def __init__(self, account=None, user=None, api_key=None, role=None, host=None):
        import snowflake.connector
        # Outside Streamlit (e.g. scripts/), fall back to .streamlit/secrets.toml.
        sf = _load_snowflake_secrets()
        account = account or sf.get("account")
        user = user or sf.get("user")
        api_key = api_key or sf.get("api_key")
        role = role or sf.get("role")
        host = host or sf.get("host")
        # Explicit args (from .streamlit/secrets.toml) win; env vars are the fallback.
        # A Snowflake programmatic access token (api_key) is accepted as the password.
        opts = dict(
            account=account or os.environ["SNOWFLAKE_ACCOUNT"],
            user=user or os.environ["SNOWFLAKE_USER"],
            password=api_key or os.environ["SNOWFLAKE_PASSWORD"],
            role=role or os.getenv("SNOWFLAKE_ROLE"),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
            database=os.getenv("SNOWFLAKE_DATABASE", "CARE_AI"),
            schema=os.getenv("SNOWFLAKE_SCHEMA", "CURATED"),
        )
        if host:
            opts["host"] = host
            opts["port"] = 443
        self.con = snowflake.connector.connect(**opts)

    def _query(self, sql, params=()):
        with self.con.cursor() as c:
            c.execute(sql, params)
            cols = [x[0].lower() for x in c.description]
            return [dict(zip(cols, r)) for r in c.fetchall()]

    def search(self, category="All", zip_code="", query="", limit=50):
        sql = """
            SELECT NPI, NAME, CATEGORY, SPECIALTY, ADDRESS, CITY, STATE, ZIP,
                   PHONE, LAST_UPDATED, AFFORDABILITY, SOURCE
            FROM CARE_AI.CURATED.TEMPE_PROVIDERS
            WHERE (%s = 'All' OR CATEGORY = %s)
              AND (%s = '' OR ZIP LIKE %s)
              AND (%s = '' OR NAME ILIKE %s OR SPECIALTY ILIKE %s)
            ORDER BY NAME
            LIMIT %s
        """
        return self._query(sql, (
            category, category,
            zip_code, f"{zip_code}%",
            query, f"%{query}%", f"%{query}%",
            int(limit),
        ))

    def search_prices(self, procedure="", limit=20):
        sql = """
            SELECT ccn, billing_code, billing_code_type, billing_code_description,
                   payer_name, rate_type, rate_amount, snapshot_date
            FROM CARE_AI.CURATED.HOSPITAL_PRICES
            WHERE (%s = '' OR billing_code_description ILIKE %s)
            ORDER BY billing_code_description, rate_amount
            LIMIT %s
        """
        return self._query(sql, (procedure, f"%{procedure}%", int(limit)))

    def get_price_comparison(self):
        sql = """
            SELECT billing_code_description, hospital_count,
                   min_price, max_price, avg_price, price_spread,
                   min_cash_price, max_cash_price
            FROM CARE_AI.CURATED.PRICE_COMPARISON
            ORDER BY price_spread DESC
            LIMIT 50
        """
        return self._query(sql)
