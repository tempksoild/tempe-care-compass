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


def _get_setting(key: str, env_var: str, default: str = "") -> str:
    """Retrieve setting from st.secrets (top-level or [snowflake]), falling back to os.environ."""
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            sf_sec = st.secrets.get("snowflake", {})
            if isinstance(sf_sec, dict):
                if key in sf_sec and sf_sec[key]:
                    return str(sf_sec[key])
                if env_var in sf_sec and sf_sec[env_var]:
                    return str(sf_sec[env_var])
            if key in st.secrets and st.secrets[key]:
                return str(st.secrets[key])
            if env_var in st.secrets and st.secrets[env_var]:
                return str(st.secrets[env_var])
    except Exception:
        pass

    val = os.environ.get(env_var, "")
    if val and val != "<none selected>":
        return val
    val_lower = os.environ.get(key, "")
    if val_lower and val_lower != "<none selected>":
        return val_lower
    return default


class SnowflakeRepository:
    def __init__(self):
        warehouse = _get_setting("warehouse", "SNOWFLAKE_WAREHOUSE", "COMPUTE_WH") or "COMPUTE_WH"
        database = _get_setting("database", "SNOWFLAKE_DATABASE", "CARE_AI") or "CARE_AI"
        schema = _get_setting("schema", "SNOWFLAKE_SCHEMA", "CURATED") or "CURATED"
        role = _get_setting("role", "SNOWFLAKE_ROLE", "ACCOUNTADMIN") or "ACCOUNTADMIN"

        self.warehouse = warehouse
        self.database = database
        self.schema = schema
        self.role = role

        self.connect_kwargs = {
            "account": _get_setting("account", "SNOWFLAKE_ACCOUNT"),
            "user": _get_setting("user", "SNOWFLAKE_USER"),
            "password": _get_setting("password", "SNOWFLAKE_PASSWORD"),
            "role": self.role,
            "warehouse": self.warehouse,
            "database": self.database,
            "schema": self.schema,
        }
        host = _get_setting("host", "SNOWFLAKE_HOST", "")
        if host:
            self.connect_kwargs["host"] = host

        self._connect()

    def _connect(self):
        import snowflake.connector
        self.con = snowflake.connector.connect(**self.connect_kwargs)
        if self.warehouse:
            try:
                with self.con.cursor() as c:
                    c.execute(f"USE WAREHOUSE {self.warehouse}")
            except Exception:
                pass

    def _query(self, sql, params=()):
        import snowflake.connector
        if hasattr(self.con, "is_closed") and self.con.is_closed():
            self._connect()

        wh = self.warehouse or "COMPUTE_WH"
        try:
            with self.con.cursor() as c:
                try:
                    c.execute(f"USE WAREHOUSE {wh}")
                except Exception:
                    pass
                c.execute(sql, params)
                cols = [x[0].lower() for x in c.description]
                return [dict(zip(cols, r)) for r in c.fetchall()]
        except Exception as e:
            if "000606" in str(e) or "No active warehouse" in str(e):
                with self.con.cursor() as c:
                    c.execute(f"USE WAREHOUSE {wh}")
                    c.execute(sql, params)
                    cols = [x[0].lower() for x in c.description]
                    return [dict(zip(cols, r)) for r in c.fetchall()]
            raise

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
