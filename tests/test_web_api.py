"""Tests for the web API logic (web/server.py) without starting an HTTP server."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web"))

from server import STATIC_FILES, ApiError, SymptomApi  # noqa: E402

from ir.search_engine import SearchEngine  # noqa: E402


@pytest.fixture(scope="module")
def api():
    return SymptomApi(SearchEngine())


def test_search_returns_ranked_results_with_disclaimer(api):
    data = api.search({"query": "high feaver with chils", "autocorrect": True})
    assert data["corrections"] == {"feaver": "fever", "chils": "chills"}
    scores = [r["score"] for r in data["results"]]
    assert len(scores) == 5 and scores == sorted(scores, reverse=True)
    assert "not a medical diagnosis" in data["disclaimer"]
    first = data["results"][0]
    assert {"disease", "precautions", "contributions", "matched_symptoms"} <= set(first)


def test_boolean_mode_returns_exact_match_set(api):
    data = api.search({"query": "feaver AND rash NOT cough", "mode": "boolean"})
    assert data["mode"] == "boolean" and data["corrections"] == {"feaver": "fever"}
    assert data["parsed"] == "AND(AND(fever, rash), NOT(cough))"
    names = {r["disease"] for r in data["results"]}
    assert {"Chicken pox", "Dengue", "Impetigo"} <= names
    assert all({"fever", "rash"} <= set(r["matched_terms"]) for r in data["results"])
    assert "not a medical diagnosis" in data["disclaimer"]
    assert "score" not in data["results"][0]          # unranked set


def test_ranked_mode_is_the_default(api):
    assert api.search({"query": "fever"})["mode"] == "ranked"


def test_search_without_autocorrect_keeps_the_query(api):
    data = api.search({"query": "feaver", "autocorrect": False})
    assert data["corrections"] == {} and data["query"] == "feaver"


def test_feedback_moves_marked_document_up(api):
    results = api.search({"query": "yellow skin and dark urine"})["results"]
    marked = results[2]["doc_id"]
    data = api.feedback({"query": "yellow skin and dark urine", "relevant": [marked],
                         "nonrelevant": [results[0]["doc_id"]], "feedback_method": "rocchio"})
    assert data["after"][0]["doc_id"] == marked and data["after"][0]["move"] == 2


@pytest.mark.parametrize("call, body", [
    ("search", {"query": ""}),
    ("search", {"query": "fever", "method": "bm25"}),
    ("search", {"query": "fever", "mode": "fuzzy"}),
    ("search", {"query": "fever AND", "mode": "boolean"}),
    ("search", {"query": "(fever OR rash", "mode": "boolean"}),
    ("feedback", {"query": "fever", "relevant": []}),
    ("feedback", {"query": "fever", "relevant": [0], "feedback_method": "magic"}),
    ("pattern", {"type": "regex", "pattern": "(["}),
    ("pattern", {"type": "nonsense", "pattern": "x"}),
])
def test_bad_requests_raise_api_error(api, call, body):
    with pytest.raises(ApiError):
        getattr(api, call)(body)


def test_lab_endpoints(api):
    assert api.pattern({"type": "wildcard", "pattern": "head*"})["matches"][:2] == ["head", "headache"]
    fuzzy = api.pattern({"type": "fuzzy", "pattern": "feaver"})
    assert fuzzy["dp"]["target"] == "fever" and fuzzy["dp"]["table"][-1][-1] == 1
    steps = api.preprocess_steps({"text": "stomach ache", "method": "stem"})["steps"]
    assert steps[-1]["value"] == ["abdomin", "pain"]
    assert api.suggest("he")["suggestions"][0] == "headache"


def test_static_files_are_a_fixed_whitelist():
    assert all(not path.startswith("/api") for path in STATIC_FILES)
    assert "/server.py" not in STATIC_FILES and "/static/../server.py" not in STATIC_FILES
