"""
Experiment 6 - Relevance feedback / query expansion.

Vectors are sparse dicts {term: weight}.  R = relevant documents, NR = non-relevant.

  Rocchio         q' = alpha*q + beta/|R| * sum_{d in R} d  -  gamma/|NR| * sum_{d in NR} d
  Ide Regular     q' = alpha*q + beta * sum_{d in R} d      -  gamma * sum_{d in NR} d
  Ide Dec-Hi      q' = alpha*q + beta * sum_{d in R} d      -  gamma * d_top_NR
                  (only the highest-ranked non-relevant document is subtracted)

Negative weights are set to 0 (a query cannot ask for "minus" a term).
Terms that appear in relevant documents but not in q are thereby ADDED to the query,
which is why relevance feedback is a form of query expansion.

Pseudo-relevance feedback (blind feedback): the top-k documents of the initial ranking are
assumed relevant (and, for the negative part, documents at lower ranks are assumed
non-relevant), so feedback can run automatically during evaluation.
"""

from ir.vector_space import normalize_vector


def _add_scaled(target, vector, scale):
    """target += scale * vector (in place)."""
    for term, weight in vector.items():
        target[term] = target.get(term, 0.0) + scale * weight


def _clip(vector, eps=1e-12):
    """Drop terms whose weight is <= 0."""
    return {t: w for t, w in vector.items() if w > eps}


def rocchio(query, relevant, nonrelevant, alpha=1.0, beta=0.75, gamma=0.15):
    """Rocchio (1971): move the query towards the centroid of R and away from that of NR."""
    new_q = {}
    _add_scaled(new_q, query, alpha)
    if relevant:
        for d in relevant:
            _add_scaled(new_q, d, beta / len(relevant))
    if nonrelevant:
        for d in nonrelevant:
            _add_scaled(new_q, d, -gamma / len(nonrelevant))
    return _clip(new_q)


def ide_regular(query, relevant, nonrelevant, alpha=1.0, beta=1.0, gamma=1.0):
    """Ide Regular: like Rocchio but sums (no averaging by |R| and |NR|)."""
    new_q = {}
    _add_scaled(new_q, query, alpha)
    for d in relevant:
        _add_scaled(new_q, d, beta)
    for d in nonrelevant:
        _add_scaled(new_q, d, -gamma)
    return _clip(new_q)


def ide_dec_hi(query, relevant, nonrelevant, alpha=1.0, beta=1.0, gamma=1.0):
    """Ide Dec-Hi: sum of relevant vectors minus ONLY the highest-ranked non-relevant one.
    `nonrelevant` must be given in rank order (best-ranked first)."""
    new_q = {}
    _add_scaled(new_q, query, alpha)
    for d in relevant:
        _add_scaled(new_q, d, beta)
    if nonrelevant:
        _add_scaled(new_q, nonrelevant[0], -gamma)
    return _clip(new_q)


METHODS = {"rocchio": rocchio, "ide_regular": ide_regular, "ide_dec_hi": ide_dec_hi}


def expand_query(query_vec, relevant, nonrelevant, method="rocchio", **params):
    """Dispatch to one of the three formulas."""
    if method not in METHODS:
        raise ValueError(f"unknown feedback method '{method}' (use {list(METHODS)})")
    return METHODS[method](query_vec, relevant, nonrelevant, **params)


def relevance_feedback(model, query_vec, relevant_ids, nonrelevant_ids, method="rocchio", **params):
    """Explicit feedback from a user.

    All vectors are length-normalised first, so alpha/beta/gamma weigh unit vectors and a
    long document does not dominate.  nonrelevant_ids must be in rank order (Dec-Hi).
    Returns the modified query vector.
    """
    q = normalize_vector(query_vec)
    rel = [model.doc_vector(i) for i in relevant_ids]
    nonrel = [model.doc_vector(i) for i in nonrelevant_ids]
    return expand_query(q, rel, nonrel, method, **params)


def pseudo_relevance_feedback(model, query, method="rocchio", k=3, nonrel_ranks=(10, 20),
                              **params):
    """Blind feedback: top-k assumed relevant, ranks nonrel_ranks[0]..nonrel_ranks[1]-1
    (0-based) assumed non-relevant.  Returns (new ranking, new query vector)."""
    query_vec = model.query_vector(query)
    initial = model.rank_vector(query_vec)
    if not initial:
        return initial, query_vec
    rel_ids = [r.doc_id for r in initial[:k]]
    nonrel_ids = [r.doc_id for r in initial[nonrel_ranks[0]:nonrel_ranks[1]]]
    new_q = relevance_feedback(model, query_vec, rel_ids, nonrel_ids, method, **params)
    return model.rank_vector(new_q), new_q


def added_terms(old_q, new_q, n=8):
    """Terms introduced by feedback, strongest first (for display)."""
    new_terms = [(t, w) for t, w in new_q.items() if t not in old_q]
    return sorted(new_terms, key=lambda x: -x[1])[:n]
