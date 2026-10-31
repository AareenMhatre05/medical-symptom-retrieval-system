import random
import re

import pytest

from ir.data_loader import clean_symptom, disease_key, load_documents
from ir.indexing import InvertedIndex, SuffixArray, SuffixTrie, build_suffix_array
from ir.porter_stemmer import PorterStemmer
from ir.preprocessing import SYNONYMS, pipeline_steps, preprocess


@pytest.fixture(scope="module")
def collection():
    return load_documents()


# --- dataset cleaning -----------------------------------------------------
def test_dataset_cleaning(collection):
    docs, severity = collection
    assert len(docs) == 41
    assert all(d.description and d.precautions for d in docs)       # every join succeeded
    assert all("_" not in s and s == s.strip() for d in docs for s in d.symptoms)
    assert all(s in severity for d in docs for s in d.symptoms)
    assert "Dimorphic hemorrhoids(piles)" in {d.disease for d in docs}
    assert clean_symptom(" dischromic _patches") == "dischromic patches"
    assert disease_key("Diabetes ") == disease_key("diabetes")


# --- preprocessing ----------------------------------------------------------
def test_synonyms_and_stopwords():
    assert preprocess("I have a stomach ache", "none") == ["abdominal", "pain"]
    assert preprocess("throwing up and high temperature", "none") == ["vomiting", "high", "fever"]


def test_no_synonym_key_inside_a_canonical_symptom(collection):
    docs, _ = collection
    symptoms = {s for d in docs for s in d.symptoms}
    for key in SYNONYMS:
        assert not any(re.search(r"\b" + re.escape(key) + r"\b", s) for s in symptoms), key


def test_stem_vs_lemma():
    assert preprocess("vomiting rashes", "stem") == ["vomit", "rash"]
    assert preprocess("vomiting rashes", "lemma") == ["vomit", "rash"]


def test_pipeline_steps_has_every_stage():
    steps = pipeline_steps("Severe chest pain, high fever")
    assert "chest pain" in " ".join(steps["8. noun groups"])
    assert steps["final terms"] == ["sever", "chest", "pain", "high", "fever"]


@pytest.mark.parametrize("word, stem", [
    ("caresses", "caress"), ("ponies", "poni"), ("hopping", "hop"), ("filing", "file"),
    ("relational", "relat"), ("conditional", "condit"), ("generalization", "gener"),
    ("happy", "happi"), ("controlling", "control"), ("vomiting", "vomit"),
])
def test_porter_examples(word, stem):
    assert PorterStemmer().stem(word) == stem


def test_porter_matches_nltk_original_algorithm(collection):
    from nltk.stem.porter import PorterStemmer as NltkPorter
    reference = NltkPorter(mode=NltkPorter.ORIGINAL_ALGORITHM)
    mine = PorterStemmer()
    docs, _ = collection
    words = {w for d in docs for w in re.findall("[a-z]+", d.text.lower())}
    assert [w for w in words if mine.stem(w) != reference.stem(w)] == []


# --- indexing -----------------------------------------------------------------
def test_inverted_index_postings():
    index = InvertedIndex([["fever", "rash", "fever"], ["cough"]])
    (posting,) = index.postings("fever")
    assert (posting.doc_id, posting.tf, posting.positions) == (0, 2, [0, 2])
    assert index.document_frequency("cough") == 1
    assert index.phrase_query(["rash", "fever"]) == {0}


def test_suffix_array_is_sorted():
    text = "banana$bandana$"
    assert build_suffix_array(text) == sorted(range(len(text)), key=lambda i: text[i:])


def test_substring_structures_agree(collection):
    docs, _ = collection
    index = InvertedIndex([preprocess(d.text, "none") for d in docs])
    sa, trie = SuffixArray(index.vocabulary), SuffixTrie(index.vocabulary)
    rng = random.Random(0)
    patterns = ["fever", "head", "itis", "rrhoe", "xyz", "a"]
    for _ in range(40):
        term = rng.choice(index.vocabulary)
        i = rng.randrange(len(term))
        patterns.append(term[i:i + rng.randint(1, 4)])
    for p in patterns:
        assert index.substring_terms(p) == sa.search(p) == trie.search(p), p
    assert sa.prefix_search("head") == {t for t in index.vocabulary if t.startswith("head")}
