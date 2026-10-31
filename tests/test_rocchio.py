import pytest

from ir.query_expansion import (ide_dec_hi, ide_regular, pseudo_relevance_feedback,
                                relevance_feedback, rocchio)
from ir.vector_space import VectorSpaceModel

Q = {"a": 1.0}
REL = [{"a": 1.0, "b": 1.0}, {"b": 1.0, "c": 1.0}]
NONREL = [{"c": 2.0}, {"b": 5.0}]


def test_rocchio_hand_computed():
    # alpha=1, beta=0.5, gamma=0.25, NR = [{c:2}]
    # a: 1 + 0.5 * (1+0)/2            = 1.25
    # b:     0.5 * (1+1)/2            = 0.5
    # c:     0.5 * (0+1)/2 - 0.25*2/1 = -0.25 -> clipped (removed)
    new_q = rocchio(Q, REL, NONREL[:1], alpha=1.0, beta=0.5, gamma=0.25)
    assert new_q == pytest.approx({"a": 1.25, "b": 0.5})


def test_rocchio_defaults_without_negatives():
    new_q = rocchio(Q, REL, [])
    # beta = 0.75: a = 1 + 0.375, b = 0.75, c = 0.375
    assert new_q == pytest.approx({"a": 1.375, "b": 0.75, "c": 0.375})


def test_ide_regular_sums_all_vectors():
    # a: 1+1 = 2 ; b: 1+1-5 = -3 -> removed ; c: 1-2 = -1 -> removed
    assert ide_regular(Q, REL, NONREL) == pytest.approx({"a": 2.0})


def test_ide_dec_hi_subtracts_only_top_nonrelevant():
    # only {c:2} (rank-1 non-relevant) is subtracted: a 2, b 2, c -1 -> removed
    assert ide_dec_hi(Q, REL, NONREL) == pytest.approx({"a": 2.0, "b": 2.0})


def test_feedback_moves_query_towards_relevant_doc(tiny_docs):
    model = VectorSpaceModel(tiny_docs, method="none")
    q = model.query_vector("fever")
    before = [r.doc_id for r in model.rank_vector(q)]
    assert set(before) == {0, 1}
    new_q = relevance_feedback(model, q, relevant_ids=[1], nonrelevant_ids=[0])
    after = [r.doc_id for r in model.rank_vector(new_q)]
    assert after[0] == 1                       # marked relevant doc now ranked first
    assert "cough" in new_q                    # query expanded with a term of doc 1


def test_pseudo_relevance_feedback_runs(tiny_docs):
    model = VectorSpaceModel(tiny_docs, method="none")
    ranking, new_q = pseudo_relevance_feedback(model, "fever rash", "rocchio", k=1)
    assert ranking[0].doc_id == 0 and new_q
