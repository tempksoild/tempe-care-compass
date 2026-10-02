# rag/retrieve.py — Candidate retrieval: lexical (BM25) + semantic, fused with RRF.
#
#   question -> detect_emergency -> parse_intent -> fresh repo query (all providers)
#            -> lexical top 50 + semantic top 50 -> reciprocal rank fusion -> pool of 40
#
# Retrieval only *finds* candidates. It does not hard-filter (rag/filter, task 4)
# or rank for the user (rag/rank.py, task 5).
#
# Semantic search: local fallback (documented in rag/README.md). There are no
# embeddings yet, so "semantic" = query expansion through SERVICE_SYNONYMS plus
# character-trigram TF-IDF cosine similarity. It catches morphology and spelling
# variants ("therapist" -> "therapy", "pediatric" -> "pediatrics") and related
# service terms ("therapist" -> "counselor", "mental health") that BM25 misses.
# Swap in Cortex EMBED_TEXT_768 + VECTOR_COSINE_SIMILARITY behind the same
# semantic_search() signature when embeddings are added.

import math
import re
from collections import Counter

from pydantic import BaseModel, Field

from rag.intent import RetrievalIntent, parse_intent

LEXICAL_TOP_K = 50
SEMANTIC_TOP_K = 50
POOL_SIZE = 40
RRF_K = 60
SEMANTIC_MIN_SIMILARITY = 0.12

# Request term -> related terms found in NPPES categories / specialties.
SERVICE_SYNONYMS: dict[str, list[str]] = {
    "therapist": ["therapy", "counselor", "psychologist", "mental health", "behavioral", "social worker"],
    "therapy": ["therapist", "counselor", "mental health", "behavioral"],
    "counseling": ["counselor", "mental health", "behavioral", "social worker"],
    "anxiety": ["mental health", "psychologist", "counselor", "psychiatric"],
    "depression": ["mental health", "psychologist", "counselor", "psychiatric"],
    "psychiatrist": ["psychiatry", "psychiatric", "mental health"],
    "addiction": ["substance use", "addiction", "behavioral"],
    "doctor": ["family", "primary care", "general practice", "internal medicine", "clinic"],
    "checkup": ["family", "primary care", "general practice"],
    "physician": ["family", "primary care", "internal medicine"],
    "pediatrician": ["pediatrics", "pediatric"],
    "kids": ["pediatrics", "pediatric"],
    "child": ["pediatrics", "pediatric"],
    "walk-in": ["urgent care", "clinic"],
    "urgent": ["urgent care"],
    "dentist": ["dental", "general practice", "dentistry"],
    "teeth": ["dental", "dentist"],
    "prescription": ["pharmacy", "community retail pharmacy"],
    "rx": ["pharmacy"],
    "orthopedic": ["orthopaedic", "orthopedic surgery", "sports medicine"],
    "orthopaedic": ["orthopedic", "sports medicine"],
    "dermatology": ["dermatologist", "skin"],
    "emergency": ["emergency medicine", "emergency care"],
    "family": ["family medicine", "family practice"],
    "mri": ["radiology", "imaging", "diagnostic"],
    "x-ray": ["radiology", "imaging"],
    "xray": ["radiology", "imaging"],
    "ultrasound": ["radiology", "imaging"],
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def provider_text(record: dict) -> str:
    return " ".join(str(record.get(k) or "") for k in ("name", "category", "specialty", "address", "city", "zip"))


def service_text(record: dict) -> str:
    """Text used for semantic matching: what the provider does, not where it is."""
    return " ".join(str(record.get(k) or "") for k in ("name", "category", "specialty"))


def _trigrams(text: str) -> Counter:
    grams = Counter()
    for tok in tokenize(text):
        padded = f"  {tok} "
        grams.update(padded[i:i + 3] for i in range(len(padded) - 2))
    return grams


class _BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = (sum(self.len) / len(docs)) if docs else 0.0
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, terms: list[str]) -> list[float]:
        out = []
        for tf, dl in zip(self.tf, self.len):
            s = 0.0
            for t in terms:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / (self.avg or 1)))
            out.append(s)
        return out


class _TrigramIndex:
    def __init__(self, texts: list[str]):
        grams = [_trigrams(t) for t in texts]
        df = Counter(g for doc in grams for g in doc)
        n = len(texts)
        self.idf = {g: math.log((1 + n) / (1 + f)) + 1 for g, f in df.items()}
        self.vecs = []
        for doc in grams:
            v = {g: c * self.idf[g] for g, c in doc.items()}
            norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
            self.vecs.append({g: x / norm for g, x in v.items()})

    def similarities(self, text: str) -> list[float]:
        q = {g: c * self.idf.get(g, 0.0) for g, c in _trigrams(text).items()}
        norm = math.sqrt(sum(x * x for x in q.values()))
        if not norm:
            return [0.0] * len(self.vecs)
        q = {g: x / norm for g, x in q.items()}
        return [sum(w * v.get(g, 0.0) for g, w in q.items()) for v in self.vecs]


class Candidate(BaseModel):
    source_id: str
    record: dict
    lexical_rank: int | None = None
    lexical_score: float | None = None
    semantic_rank: int | None = None
    semantic_score: float | None = None
    rrf_score: float = 0.0


class RetrievalResult(BaseModel):
    intent: RetrievalIntent
    candidates: list[Candidate] = Field(default_factory=list)
    lexical_terms: list[str] = Field(default_factory=list)
    semantic_query: str = ""
    lexical_hits: int = 0
    semantic_hits: int = 0
    corpus_size: int = 0
    short_circuited: bool = False
    note: str | None = None


def lexical_terms(intent: RetrievalIntent) -> list[str]:
    """Keywords + category words + ZIP. Category words bias toward the requested
    category without hard-filtering it (that is the eligibility stage)."""
    terms = list(intent.keywords)
    if intent.category != "All":
        terms += tokenize(intent.category)
    if intent.zip_code:
        terms.append(intent.zip_code)
    return list(dict.fromkeys(t for t in (tok for term in terms for tok in tokenize(term))))


def semantic_query(intent: RetrievalIntent) -> str:
    parts = list(intent.keywords)
    for kw in intent.keywords:
        parts += SERVICE_SYNONYMS.get(kw, [])
    if intent.category_term:
        parts += SERVICE_SYNONYMS.get(intent.category_term, [])
    if intent.category != "All":
        parts.append(intent.category)
    return " ".join(dict.fromkeys(parts))


def reciprocal_rank_fusion(*rankings: list[str], k: int = RRF_K) -> list[tuple[str, float]]:
    """rankings: lists of source_ids, best first. Returns (source_id, score), best first.
    Ties break by source_id so the order is deterministic."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for pos, sid in enumerate(ranking, start=1):
            scores[sid] = scores.get(sid, 0.0) + 1.0 / (k + pos)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class ProviderRetriever:
    """Runs a fresh repository query per request; never reuses sidebar rows.

    The provider corpus is loaded from `repo.all_providers()` and indexed once per
    retriever instance (pass refresh=True to reload)."""

    def __init__(self, repo, lexical_k=LEXICAL_TOP_K, semantic_k=SEMANTIC_TOP_K, pool_size=POOL_SIZE):
        self.repo = repo
        self.lexical_k, self.semantic_k, self.pool_size = lexical_k, semantic_k, pool_size
        self._records: list[dict] | None = None

    def _load(self, refresh: bool = False):
        if self._records is None or refresh:
            self._records = self.repo.all_providers()
            self.__dict__.pop("_by_id", None)
            self._bm25 = _BM25([tokenize(provider_text(r)) for r in self._records])
            self._tri = _TrigramIndex([service_text(r) for r in self._records])

    def record(self, source_id: str) -> dict | None:
        """Raw provider record (address/city/zip for maps) by source_id."""
        self._load()
        if not hasattr(self, "_by_id"):
            self._by_id = {r["source_id"]: r for r in self._records}
        return self._by_id.get(source_id)

    def lexical_search(self, terms: list[str], top_k: int) -> list[tuple[int, float]]:
        if not terms:
            return []
        scored = [(i, s) for i, s in enumerate(self._bm25.scores(terms)) if s > 0]
        scored.sort(key=lambda x: (-x[1], self._records[x[0]]["source_id"]))
        return scored[:top_k]

    def semantic_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        if not query.strip():
            return []
        scored = [(i, s) for i, s in enumerate(self._tri.similarities(query)) if s >= SEMANTIC_MIN_SIMILARITY]
        scored.sort(key=lambda x: (-x[1], self._records[x[0]]["source_id"]))
        return scored[:top_k]

    def retrieve(self, intent: RetrievalIntent, refresh: bool = False) -> RetrievalResult:
        if intent.emergency:
            return RetrievalResult(intent=intent, short_circuited=True,
                                   note=f"Emergency detected ({intent.emergency_reason}); retrieval skipped.")
        self._load(refresh)
        terms = lexical_terms(intent)
        sem_q = semantic_query(intent)
        lex = self.lexical_search(terms, self.lexical_k)
        sem = self.semantic_search(sem_q, self.semantic_k)

        recs = self._records
        lex_ids = [recs[i]["source_id"] for i, _ in lex]
        sem_ids = [recs[i]["source_id"] for i, _ in sem]
        by_id = {r["source_id"]: r for r in recs}
        lex_meta = {recs[i]["source_id"]: (pos, s) for pos, (i, s) in enumerate(lex, 1)}
        sem_meta = {recs[i]["source_id"]: (pos, s) for pos, (i, s) in enumerate(sem, 1)}

        pool = []
        for sid, score in reciprocal_rank_fusion(lex_ids, sem_ids)[: self.pool_size]:
            lr, ls = lex_meta.get(sid, (None, None))
            sr, ss = sem_meta.get(sid, (None, None))
            pool.append(Candidate(source_id=sid, record=by_id[sid], lexical_rank=lr, lexical_score=ls,
                                  semantic_rank=sr, semantic_score=ss, rrf_score=score))
        return RetrievalResult(
            intent=intent, candidates=pool, lexical_terms=terms, semantic_query=sem_q,
            lexical_hits=len(lex), semantic_hits=len(sem), corpus_size=len(recs),
            note=None if pool else "No provider text matched the request terms.",
        )


def retrieve_for_question(question: str, retriever: ProviderRetriever) -> RetrievalResult:
    """Safety detection -> intent parsing -> fresh retrieval."""
    return retriever.retrieve(parse_intent(question))
