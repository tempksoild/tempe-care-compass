# rag/prices.py — Comparable price matching + price_score (task 10).
#
# Prices are only compared when records share billing code, code type, care setting,
# rate (price) type and compatible snapshot dates, and the hospital (CCN) is known.
#   price_score = 100 * (max - price) / (max - min);  all equal -> 100
# Non-local hospitals are price_eligible=False (score None): the Healthparse sample
# is California (CCN state code 05) and must not rank Tempe providers.

from datetime import date

from pydantic import BaseModel, Field

from rag import config as cfg
from rag.citations import normalize_ccn, price_source_id


class HospitalPrice(BaseModel):
    ccn: str
    price: float
    price_score: float | None
    price_eligible: bool
    group: tuple[str, str, str, str]       # (code_type, code, setting, rate_type)
    group_min: float
    group_max: float
    hospitals_compared: int
    source_ids: list[str] = Field(default_factory=list)
    note: str = ""


def is_local_ccn(ccn) -> bool:
    return normalize_ccn(ccn)[:2] in cfg.LOCAL_CCN_STATE_CODES


def _date(v):
    try:
        return date.fromisoformat(str(v)[:10])
    except ValueError:
        return None


def comparable_rows(rows: list[dict], billing_codes: list[tuple[str, str]]) -> dict[tuple, list[dict]]:
    """Group rows that may be compared. Drops: other codes, unknown setting, unknown CCN,
    non-positive prices, exact duplicates, snapshots too far from the group's latest."""
    wanted = {(t.upper(), str(c)) for t, c in billing_codes}
    groups: dict[tuple, list[dict]] = {}
    seen = set()
    for r in rows:
        ctype, code = str(r.get("billing_code_type") or "").upper(), str(r.get("billing_code") or "").strip()
        setting = str(r.get("setting") or "").strip().lower()
        rtype = str(r.get("rate_type") or "").strip().lower()
        ccn = normalize_ccn(r.get("ccn"))
        try:
            amount = float(r.get("rate_amount"))
        except (TypeError, ValueError):
            continue
        if (ctype, code) not in wanted or not setting or ccn == "unknown" or amount <= 0 or not rtype:
            continue
        sid = price_source_id(r)
        if sid in seen:
            continue
        seen.add(sid)
        groups.setdefault((ctype, code, setting, rtype), []).append({**r, "ccn": ccn, "rate_amount": amount,
                                                                    "source_id": sid})
    for key, members in groups.items():
        dates = [d for d in (_date(m.get("snapshot_date")) for m in members) if d]
        if dates:
            latest = max(dates)
            groups[key] = [m for m in members if (d := _date(m.get("snapshot_date"))) and
                           (latest - d).days <= cfg.PRICE_DATE_TOLERANCE_DAYS]
    return {k: v for k, v in groups.items() if v}


def pick_group(groups: dict[tuple, list[dict]]) -> tuple | None:
    """Preferred rate type first (cash/self-pay), then the group covering most hospitals."""
    def hospitals(k):
        return len({m["ccn"] for m in groups[k]})
    eligible = [k for k in groups if hospitals(k) >= cfg.PRICE_MIN_HOSPITALS]
    if not eligible:
        return None
    pref = {t: i for i, t in enumerate(cfg.PRICE_RATE_TYPE_PREFERENCE)}
    return sorted(eligible, key=lambda k: (pref.get(k[3], len(pref)), -hospitals(k), k))[0]


def hospital_price_scores(rows: list[dict], billing_codes: list[tuple[str, str]]) -> dict[str, HospitalPrice]:
    """-> {ccn: HospitalPrice}. Each hospital's price = its lowest rate in the chosen group."""
    groups = comparable_rows(rows, billing_codes)
    key = pick_group(groups)
    if key is None:
        return {}
    by_ccn: dict[str, list[dict]] = {}
    for m in groups[key]:
        by_ccn.setdefault(m["ccn"], []).append(m)
    prices = {ccn: min(ms, key=lambda m: m["rate_amount"]) for ccn, ms in by_ccn.items()}
    lo, hi = min(p["rate_amount"] for p in prices.values()), max(p["rate_amount"] for p in prices.values())
    out = {}
    for ccn, best in prices.items():
        local = is_local_ccn(ccn)
        score = None
        if local:
            score = 100.0 if hi == lo else round(100 * (hi - best["rate_amount"]) / (hi - lo), 1)
        out[ccn] = HospitalPrice(
            ccn=ccn, price=best["rate_amount"], price_score=score, price_eligible=local, group=key,
            group_min=lo, group_max=hi, hospitals_compared=len(prices), source_ids=[best["source_id"]],
            note="" if local else "Out-of-state hospital price (not used for Tempe ranking)",
        )
    return out


def build_price_index(candidates, rows: list[dict]) -> dict[str, set]:
    """Provider source_id -> {(code_type, code)} for candidates whose record has a CCN
    with price rows. Feeds EligibilityFilter(price_index=...)."""
    by_ccn: dict[str, set] = {}
    for r in rows:
        by_ccn.setdefault(normalize_ccn(r.get("ccn")), set()).add(
            (str(r.get("billing_code_type") or "").upper(), str(r.get("billing_code") or "").strip()))
    return {c.source_id: by_ccn.get(normalize_ccn(c.record.get("ccn")), set())
            for c in candidates if c.record.get("ccn")}
