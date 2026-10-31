"""Hand-computed example:
    ranked   = [1, 2, 3, 4, 5]
    relevant = {1, 3, 6}           (doc 6 is never retrieved)
    grades   = {1: 2, 3: 1, 6: 2}
"""
import math

import pytest

from ir.evaluation import (average_precision, dcg_at_k, evaluate_query, evaluate_run, f1_score,
                           interpolated_precision, mean_average_precision, ndcg_at_k,
                           precision, precision_at_k, recall, reciprocal_rank)

RANKED = [1, 2, 3, 4, 5]
RELEVANT = {1, 3, 6}
GRADES = {1: 2, 3: 1, 6: 2}


def test_precision_recall_f1():
    assert precision(RANKED, RELEVANT) == pytest.approx(2 / 5)
    assert recall(RANKED, RELEVANT) == pytest.approx(2 / 3)
    # F1 = 2 * 0.4 * 0.6667 / (0.4 + 0.6667) = 0.5
    assert f1_score(RANKED, RELEVANT) == pytest.approx(0.5)
    assert precision([], RELEVANT) == 0.0 and f1_score([], RELEVANT) == 0.0


def test_precision_at_k():
    assert precision_at_k(RANKED, RELEVANT, 1) == 1.0
    assert precision_at_k(RANKED, RELEVANT, 3) == pytest.approx(2 / 3)
    assert precision_at_k(RANKED, RELEVANT, 5) == pytest.approx(2 / 5)
    assert precision_at_k([1], RELEVANT, 5) == pytest.approx(1 / 5)  # missing ranks = 0


def test_average_precision_and_map():
    # relevant at ranks 1 and 3: (1/1 + 2/3) / 3 relevant docs = 0.5556
    assert average_precision(RANKED, RELEVANT) == pytest.approx((1 + 2 / 3) / 3)
    # second query: relevant doc at rank 2 -> AP = 0.5
    assert average_precision([2, 1], {1}) == pytest.approx(0.5)
    assert mean_average_precision([(RANKED, RELEVANT), ([2, 1], {1})]) == pytest.approx(
        ((1 + 2 / 3) / 3 + 0.5) / 2)


def test_reciprocal_rank():
    assert reciprocal_rank(RANKED, RELEVANT) == 1.0
    assert reciprocal_rank([9, 8, 3], RELEVANT) == pytest.approx(1 / 3)
    assert reciprocal_rank([9], RELEVANT) == 0.0


def test_dcg_and_ndcg():
    # DCG@3  = (2^2-1)/log2(2) + 0 + (2^1-1)/log2(4)       = 3 + 0.5 = 3.5
    # IDCG@3 = 3/log2(2) + 3/log2(3) + 1/log2(4)            = 3 + 1.8928 + 0.5
    assert dcg_at_k(RANKED, GRADES, 3) == pytest.approx(3.5)
    idcg = 3 + 3 / math.log2(3) + 0.5
    assert ndcg_at_k(RANKED, GRADES, 3) == pytest.approx(3.5 / idcg)
    assert ndcg_at_k([1, 6, 3], GRADES, 3) == pytest.approx(1.0)   # ideal order
    assert ndcg_at_k(RANKED, {}, 3) == 0.0


def test_interpolated_precision():
    # (recall, precision) after each rank: (1/3,1) (1/3,.5) (2/3,.667) (2/3,.5) (2/3,.4)
    p = interpolated_precision(RANKED, RELEVANT, levels=[0.0, 0.3, 0.4, 0.6, 0.7])
    assert p == pytest.approx([1.0, 1.0, 2 / 3, 2 / 3, 0.0])


def test_evaluate_query_and_run():
    m = evaluate_query(RANKED, GRADES, cutoff=5)
    assert m["P"] == pytest.approx(0.4) and m["P@1"] == 1.0
    assert m["AP"] == pytest.approx((1 + 2 / 3) / 3)
    summary, per_query = evaluate_run([RANKED, [2, 1]], [GRADES, {1: 2}])
    assert len(per_query) == 2
    assert summary["MAP"] == pytest.approx(((1 + 2 / 3) / 3 + 0.5) / 2)
    assert summary["MRR"] == pytest.approx((1 + 0.5) / 2)
