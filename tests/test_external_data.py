import os

import pytest

from ir.data_loader import disease_key, load_documents
from ir.external_data import (BROADER_TOPICS, MEDLINEPLUS_SUBSET, MEDLINEPLUS_TOPICS, S2D_DIR,
                              enrich_documents,
                              expansion_texts, load_medlineplus, load_symptom2disease)

HAS_S2D = os.path.exists(os.path.join(S2D_DIR, "train.jsonl"))
HAS_MEDLINEPLUS = os.path.exists(MEDLINEPLUS_SUBSET)


@pytest.fixture(scope="module")
def docs():
    return load_documents()[0]


def test_every_disease_has_a_medlineplus_topic(docs):
    assert set(MEDLINEPLUS_TOPICS) == {d.disease for d in docs}


def test_document_text_without_enrichment_is_unchanged(docs):
    doc = docs[0]
    assert doc.text == doc.core_text
    assert doc.text.startswith(doc.disease + ".") and "Also called" not in doc.text


def test_enrichment_returns_copies(docs):
    enriched, _ = enrich_documents(docs, medlineplus=True, symptom2disease=True)
    assert all(not d.also_called and not d.user_descriptions for d in docs)  # originals intact
    for d in enriched:
        assert d.user_descriptions == [] or d.text.startswith(d.core_text)
        assert all(text not in d.core_text for text in d.user_descriptions)


@pytest.mark.skipif(not HAS_MEDLINEPLUS, reason="MedlinePlus subset not downloaded")
def test_medlineplus_names(docs):
    names = load_medlineplus()
    assert set(names) == {d.disease for d in docs}
    enriched, info = enrich_documents(docs, medlineplus=True, symptom2disease=False)
    by_name = {d.disease: d for d in enriched}
    assert "Piles" in by_name["Dimorphic hemorrhoids(piles)"].also_called
    assert "Stomach flu" in by_name["Gastroenteritis"].also_called
    # broader topics (Typhoid -> "Salmonella Infections") contribute no names
    assert by_name["Typhoid"].also_called == [] and by_name["Hepatitis E"].also_called == []
    assert info["medlineplus_docs"] == 41 - len(BROADER_TOPICS)


@pytest.mark.skipif(not HAS_S2D, reason="Symptom2Disease not downloaded")
def test_symptom2disease_labels_and_no_test_leakage(docs):
    keys = {disease_key(d.disease) for d in docs}
    train, test = load_symptom2disease("train"), load_symptom2disease("test")
    assert len(train) == 853 and len(test) == 212
    assert {disease_key(label) for _, label in train + test} <= keys  # every label maps to a doc
    test_texts = {t.lower() for t, _ in test}
    expansion = [t for texts in expansion_texts().values() for t in texts]
    assert expansion and not any(t.lower() in test_texts for t in expansion)
