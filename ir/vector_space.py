"""
Experiment 4 - Vector space model (the main ranking engine).

Weights (all computed from scratch, base-10 logarithms):

    tf-weight   w_tf(t, d) = 1 + log10(tf(t, d))        if tf > 0, else 0
    idf         idf(t)     = log10(N / df(t))
    tf-idf      w(t, d)    = w_tf(t, d) * idf(t)

Documents and queries are sparse vectors {term: weight}.  Ranking uses cosine similarity

    cos(q, d) = sum_t q_t * d_t / (|q| * |d|),     |v| = sqrt(sum_t v_t^2)

Only documents in the posting lists of the query terms (inverted index) are scored.

Optional severity boost (Symptom-severity.csv, weights 1-7): if the query mentions a
known symptom (all terms of the symptom name are in the query), the query weight of
each of that symptom's terms is multiplied by

    1 + lambda * severity / 7

so that, e.g., "chest pain" (7) counts more than "itching" (1).
"""

import math
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

from ir.indexing import InvertedIndex
from ir.preprocessing import preprocess

MAX_SEVERITY = 7


# ---------------------------------------------------------------------------
# Weighting functions
# ---------------------------------------------------------------------------
def term_frequencies(tokens):
    """Raw counts tf(t, d)."""
    return dict(Counter(tokens))


def tf_weight(count):
    """Log-scaled tf: 1 + log10(tf) for tf > 0, otherwise 0."""
    return 1.0 + math.log10(count) if count > 0 else 0.0


def inverse_document_frequency(df, num_docs):
    """idf = log10(N / df).  A term found in every document gets idf 0."""
    return math.log10(num_docs / df) if df > 0 else 0.0


def tfidf_vector(tokens, idf):
    """{term: (1 + log10 tf) * idf}; terms unknown to the collection are dropped."""
    return {t: tf_weight(c) * idf[t] for t, c in term_frequencies(tokens).items()
            if idf.get(t, 0.0) > 0}


def vector_norm(vector):
    """Euclidean length |v| = sqrt(sum v_t^2)."""
    return math.sqrt(sum(w * w for w in vector.values()))


def dot_product(v1, v2):
    """sum_t v1_t * v2_t over shared terms (iterate over the shorter vector)."""
    if len(v1) > len(v2):
        v1, v2 = v2, v1
    return sum(w * v2[t] for t, w in v1.items() if t in v2)


def cosine_similarity(v1, v2):
    """cos(v1, v2) = v1 . v2 / (|v1| |v2|); 0 if either vector is empty."""
    n1, n2 = vector_norm(v1), vector_norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot_product(v1, v2) / (n1 * n2)


def normalize_vector(vector):
    """v / |v| (unit length)."""
    n = vector_norm(vector)
    return {t: w / n for t, w in vector.items()} if n else {}


@dataclass
class SearchResult:
    doc_id: int
    score: float
    contributions: list = field(default_factory=list)  # [(term, share of the cosine)]

    @property
    def matched_terms(self):
        return [t for t, _ in self.contributions]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class VectorSpaceModel:
    """TF-IDF vector space model over the disease documents."""

    def __init__(self, documents, method="stem", severity=None, severity_boost=0.0):
        """
        documents       list of data_loader.Document
        method          'stem' | 'lemma' | 'none'   (preprocessing normaliser)
        severity        {symptom name: weight 1-7} (from data_loader) or None
        severity_boost  lambda in the boost factor; 0 disables the boost
        """
        self.documents = documents
        self.method = method
        self.severity_boost = severity_boost
        self.doc_tokens = [preprocess(d.text, method) for d in documents]
        self.index = InvertedIndex(self.doc_tokens)
        self.num_docs = len(documents)
        self.idf = {t: inverse_document_frequency(len(p), self.num_docs)
                    for t, p in self.index.index.items()}
        self.doc_vectors = [tfidf_vector(tokens, self.idf) for tokens in self.doc_tokens]
        self.doc_norms = [vector_norm(v) for v in self.doc_vectors]
        # symptom name -> its index terms, used for the severity boost and for display
        self.symptom_terms = {}
        for doc in documents:
            for symptom in doc.symptoms:
                if symptom not in self.symptom_terms:
                    self.symptom_terms[symptom] = preprocess(symptom, method)
        self.severity = severity or {}

    # -- query side -------------------------------------------------------
    def detect_symptoms(self, query_terms):
        """Known symptoms whose terms all occur in the query."""
        terms = set(query_terms)
        return [s for s, ts in self.symptom_terms.items() if ts and set(ts) <= terms]

    def severity_factors(self, query_terms):
        """{term: 1 + lambda * severity/7} for terms of symptoms mentioned in the query."""
        factors = {}
        if self.severity_boost <= 0:
            return factors
        for symptom in self.detect_symptoms(query_terms):
            weight = self.severity.get(symptom)
            if weight is None:
                continue
            factor = 1.0 + self.severity_boost * weight / MAX_SEVERITY
            for term in self.symptom_terms[symptom]:
                factors[term] = max(factors.get(term, 1.0), factor)
        return factors

    def query_vector(self, query, preprocessed=False):
        """Query text -> tf-idf vector (with severity boost when enabled)."""
        terms = query if preprocessed else preprocess(query, self.method)
        vector = tfidf_vector(terms, self.idf)
        for term, factor in self.severity_factors(terms).items():
            if term in vector:
                vector[term] *= factor
        return vector

    # -- document side ----------------------------------------------------
    def doc_vector(self, doc_id, normalized=True):
        vector = self.doc_vectors[doc_id]
        return normalize_vector(vector) if normalized else dict(vector)

    # -- ranking ----------------------------------------------------------
    def rank_vector(self, query_vec, top_k=None, use_index=True):
        """Rank documents by cosine similarity with an arbitrary query vector.

        Score breakdown: contribution(t) = q_t * d_t / (|q| |d|); contributions sum to
        the cosine score.  Documents with score 0 are not returned.
        """
        q_norm = vector_norm(query_vec)
        if q_norm == 0:
            return []
        positive_terms = [t for t, w in query_vec.items() if w > 0]
        candidates = (self.index.candidates(positive_terms) if use_index
                      else range(self.num_docs))
        results = []
        for doc_id in candidates:
            d_vec, d_norm = self.doc_vectors[doc_id], self.doc_norms[doc_id]
            if d_norm == 0:
                continue
            contributions = [(t, w * d_vec[t] / (q_norm * d_norm))
                             for t, w in query_vec.items() if t in d_vec]
            score = sum(c for _, c in contributions)
            if score > 0:
                contributions.sort(key=lambda x: -x[1])
                results.append(SearchResult(doc_id, score, contributions))
        results.sort(key=lambda r: (-r.score, r.doc_id))
        return results[:top_k] if top_k else results

    def search(self, query, top_k=None):
        """Free-text query -> ranked list of SearchResult."""
        return self.rank_vector(self.query_vector(query), top_k)

    # -- explanation helpers -----------------------------------------------
    def matched_symptoms(self, doc_id, query_terms):
        """Symptoms of the document that share at least one term with the query,
        fully matched symptoms first."""
        terms = set(query_terms)
        full, partial = [], []
        for symptom in self.documents[doc_id].symptoms:
            st = set(self.symptom_terms.get(symptom, []))
            if st and st <= terms:
                full.append(symptom)
            elif st & terms:
                partial.append(symptom)
        return full, partial

    def tfidf_matrix(self):
        """Dense term x document TF-IDF matrix as a DataFrame (for inspection/export)."""
        names = [d.disease for d in self.documents]
        vocab = sorted(self.idf)
        data = {name: [vec.get(t, 0.0) for t in vocab] for name, vec in zip(names, self.doc_vectors)}
        return pd.DataFrame(data, index=vocab)
