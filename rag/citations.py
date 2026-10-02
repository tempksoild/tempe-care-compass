# rag/citations.py — Stable, deterministic source IDs for provider and price records.
#
# IDs are derived from record content (not row position), so the same record
# always gets the same ID across reloads and between CSV and Snowflake.
#   Provider: NPPES:<npi>:address:<hash8>
#   Price:    PRICE:<ccn>:<code_type>:<code>:<rate_type>:<hash8>
#   User:     USER_LOCATION:<method>:request   (internal only, not a public citation)

import hashlib
import re


def _norm(value) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = re.sub(r"[^a-z0-9 ]", " ", str(value or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def _hash8(*parts) -> str:
    joined = "|".join(_norm(p) for p in parts)
    return hashlib.sha1(joined.encode()).hexdigest()[:8]


def normalized_location(record: dict) -> str:
    """Location key used for de-duplication (address + zip, normalized)."""
    return f"{_norm(record.get('address'))}|{str(record.get('zip') or '')[:5]}"


def provider_source_id(record: dict) -> str:
    npi = str(record.get("npi") or "unknown").strip()
    return f"NPPES:{npi}:address:{_hash8(record.get('address'), str(record.get('zip') or '')[:5])}"


def normalize_ccn(value) -> str:
    """CMS Certification Numbers are 6 characters; restore leading zeros lost by numeric parsing."""
    ccn = str(value or "").strip()
    if ccn.endswith(".0"):
        ccn = ccn[:-2]
    return ccn.zfill(6) if ccn.isdigit() else (ccn or "unknown")


def price_source_id(record: dict) -> str:
    h = _hash8(record.get("payer_name"), record.get("rate_amount"), record.get("snapshot_date"))
    return ":".join([
        "PRICE",
        normalize_ccn(record.get("ccn")),
        str(record.get("billing_code_type") or "").strip(),
        str(record.get("billing_code") or "").strip(),
        str(record.get("rate_type") or "").strip(),
        h,
    ])


def user_location_id(method: str) -> str:
    return f"USER_LOCATION:{method}:request"


def attach_source_ids(records: list[dict], kind: str = "provider") -> list[dict]:
    """Return copies of records with a `source_id` field added."""
    fn = provider_source_id if kind == "provider" else price_source_id
    return [{**r, "source_id": r.get("source_id") or fn(r)} for r in records]
