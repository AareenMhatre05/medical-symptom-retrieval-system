import math

import pytest

from ir.vector_space import (VectorSpaceModel, cosine_similarity, inverse_document_frequency,
                             tf_weight, tfidf_vector, vector_norm)


def test_cosine_hand_example():
    # (1,2).(2,1) = 4 ; |v1| = |v2| = sqrt(5)  ->  4/5
    assert cosine_similarity({"x": 1, "y": 2}, {"x": 2, "y": 1}) == pytest.approx(0.8)


def test_cosine_identical_orthogonal_and_empty():
    assert cosine_similarity({"a": 3, "b": 4}, {"a": 3, "b": 4}) == pytest.approx(1.0)
    assert cosine_similarity({"a": 1}, {"b": 1}) == 0.0
    assert cosine_similarity({}, {"a": 1}) == 0.0


def test_cosine_is_length_invariant():
    assert cosine_similarity({"a": 1, "b": 1}, {"a": 10, "b": 10}) == pytest.approx(1.0)


def test_weighting_formulas():
    assert tf_weight(0) == 0.0
    assert tf_weight(1) == 1.0
    assert tf_weight(10) == pytest.approx(2.0)                 # 1 + log10(10)
    assert inverse_document_frequency(10, 100) == pytest.approx(1.0)
    assert inverse_document_frequency(100, 100) == 0.0
    vec = tfidf_vector(["a", "a", "b"], {"a": 0.5, "b": 1.0})
    assert vec["a"] == pytest.approx((1 + math.log10(2)) * 0.5)
    assert vec["b"] == pytest.approx(1.0)
    assert vector_norm({"a": 3, "b": 4}) == 5


def test_vsm_ranking_on_tiny_collection(tiny_docs):
    model = VectorSpaceModel(tiny_docs, method="none")
    # idf(fever) = idf(rash) = log10(3/2); idf(cough) = idf(itching) = log10(3)
    results = model.search("fever rash")
    assert results[0].doc_id == 0                     # only doc with both terms
    assert {r.doc_id for r in results} == {0, 1, 2}
    # contributions sum to the cosine score
    for r in results:
        assert sum(c for _, c in r.contributions) == pytest.approx(r.score)
    q = model.query_vector("fever rash")
    assert results[0].score == pytest.approx(cosine_similarity(q, model.doc_vectors[0]))


def test_vsm_unknown_query_returns_nothing(tiny_docs):
    assert VectorSpaceModel(tiny_docs, method="none").search("toothache") == []
