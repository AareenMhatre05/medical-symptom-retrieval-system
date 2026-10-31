"""
Experiment 8 - Evaluation metrics, implemented from scratch.

Notation: `ranked` is the ranked list of retrieved doc ids, `relevant` the set of relevant
doc ids, `grades` a dict doc_id -> graded relevance (2 = correct, 1 = plausible, 0 = no).
Binary metrics treat a document as relevant when grade >= threshold (default 1).

  Precision      P  = |Ret ∩ Rel| / |Ret|
  Recall         R  = |Ret ∩ Rel| / |Rel|
  F1             F1 = 2PR / (P + R)
  Precision@K    P@K = |top-K ∩ Rel| / K
  Average Prec.  AP = (1/|Rel|) * sum_{k : ranked[k] in Rel} P@k
  MAP            mean of AP over all queries
  DCG@K          sum_{i=1..K} (2^g_i - 1) / log2(i + 1)
  NDCG@K         DCG@K / IDCG@K    (IDCG = DCG of the ideal ordering of the judgments)
  RR             1 / rank of the first relevant document (MRR = mean over queries)
"""

import math


def precision(retrieved, relevant):
    retrieved = set(retrieved)
    if not retrieved:
        return 0.0
    return len(retrieved & set(relevant)) / len(retrieved)


def recall(retrieved, relevant):
    relevant = set(relevant)
    if not relevant:
        return 0.0
    return len(set(retrieved) & relevant) / len(relevant)


def f1_score(retrieved, relevant):
    p, r = precision(retrieved, relevant), recall(retrieved, relevant)
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def precision_at_k(ranked, relevant, k):
    """Missing positions (fewer than k results) count as non-relevant."""
    relevant = set(relevant)
    return sum(1 for d in ranked[:k] if d in relevant) / k


def average_precision(ranked, relevant):
    relevant = set(relevant)
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for i, doc in enumerate(ranked, start=1):
        if doc in relevant:
            hits += 1
            total += hits / i  # precision at this rank
    return total / len(relevant)


def mean_average_precision(runs):
    """runs: list of (ranked, relevant) pairs, one per query."""
    if not runs:
        return 0.0
    return sum(average_precision(r, rel) for r, rel in runs) / len(runs)


def reciprocal_rank(ranked, relevant):
    relevant = set(relevant)
    for i, doc in enumerate(ranked, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def dcg_at_k(ranked, grades, k):
    """DCG@K with exponential gain (2^g - 1) and log2(i + 1) discount (i = 1-based rank)."""
    return sum((2 ** grades.get(doc, 0) - 1) / math.log2(i + 1)
               for i, doc in enumerate(ranked[:k], start=1))


def ndcg_at_k(ranked, grades, k):
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg_at_k(ranked, grades, k) / idcg if idcg > 0 else 0.0


def precision_recall_points(ranked, relevant):
    """(recall, precision) after each retrieved document."""
    relevant = set(relevant)
    points, hits = [], 0
    for i, doc in enumerate(ranked, start=1):
        if doc in relevant:
            hits += 1
        points.append((hits / len(relevant) if relevant else 0.0, hits / i))
    return points


def interpolated_precision(ranked, relevant, levels=None):
    """11-point interpolated precision: P_interp(r) = max_{r' >= r} P(r')."""
    levels = levels if levels is not None else [i / 10 for i in range(11)]
    points = precision_recall_points(ranked, relevant)
    return [max((p for r, p in points if r >= level - 1e-12), default=0.0) for level in levels]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def relevant_set(grades, threshold=1):
    return {d for d, g in grades.items() if g >= threshold}


def evaluate_query(ranked, grades, ks=(1, 3, 5), cutoff=5, threshold=1):
    """All metrics for one query.  P / R / F1 use the top-`cutoff` results as the retrieved
    set (what the user sees); for an unranked Boolean result pass cutoff=None."""
    relevant = relevant_set(grades, threshold)
    retrieved = ranked if cutoff is None else ranked[:cutoff]
    result = {
        "P": precision(retrieved, relevant),
        "R": recall(retrieved, relevant),
        "F1": f1_score(retrieved, relevant),
    }
    for k in ks:
        result[f"P@{k}"] = precision_at_k(ranked, relevant, k)
    result["AP"] = average_precision(ranked, relevant)
    result["RR"] = reciprocal_rank(ranked, relevant)
    for k in ks:
        if k > 1:
            result[f"NDCG@{k}"] = ndcg_at_k(ranked, grades, k)
    result["retrieved"] = len(ranked)
    return result


def evaluate_run(ranked_lists, judgments, **kwargs):
    """Average every metric over the queries.  AP averages to MAP and RR to MRR."""
    per_query = [evaluate_query(r, g, **kwargs) for r, g in zip(ranked_lists, judgments)]
    keys = per_query[0].keys()
    summary = {k: sum(q[k] for q in per_query) / len(per_query) for k in keys}
    summary["MAP"] = summary.pop("AP")
    summary["MRR"] = summary.pop("RR")
    return summary, per_query
