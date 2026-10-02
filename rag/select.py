# rag/select.py — Messages for the selection step.
# Only the zero-result case is implemented here; A/B/C selection rules (exactly 3,
# 2/1 handling) are still open in task 6.

from rag.filter import CONSTRAINT_LABELS


def no_eligible_message(eliminated_by: dict[str, int], considered: int, retrieval_note: str | None = None) -> str:
    """Explain why zero providers are returned, naming the eliminating constraint(s)."""
    if considered == 0:
        reason = retrieval_note or "No provider records matched the search terms."
        return f"No providers met the requirements. {reason} Try broader terms, or call 211 for resource navigation."
    parts = [f"{n} of {considered} by {CONSTRAINT_LABELS.get(k, k)}"
             for k, n in sorted(eliminated_by.items(), key=lambda kv: (-kv[1], kv[0]))]
    top = max(eliminated_by.items(), key=lambda kv: (kv[1], kv[0]))[0] if eliminated_by else None
    msg = f"No providers met the requirements. Candidates were removed: {'; '.join(parts)}."
    if top:
        msg += f" The main constraint was: {CONSTRAINT_LABELS.get(top, top)}."
    return msg
