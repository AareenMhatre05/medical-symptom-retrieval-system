"""
Facade that wires the experiments together for main.py and eval/run_evaluation.py:
cleaned documents (optionally enriched with MedlinePlus names and Symptom2Disease
document expansion), Boolean model, vector space models (built lazily per configuration),
spelling correction and symptom suggestions.
"""

from collections import Counter

from ir.boolean_model import BooleanModel
from ir.data_loader import DEFAULT_DATA_DIR, disease_key, load_documents
from ir.external_data import enrich_documents
from ir.pattern_matching import PatternMatcher, suggest_symptoms
from ir.preprocessing import (STOPWORDS, SYNONYMS, lowercase, remove_punctuation,
                              tokenize)
from ir.vector_space import VectorSpaceModel

DEFAULT_SEVERITY_BOOST = 0.5


class SearchEngine:
    def __init__(self, data_dir=DEFAULT_DATA_DIR, method="stem", medlineplus=True,
                 symptom2disease=True):
        """medlineplus / symptom2disease: use the external sources when their files exist
        (pass False for both to search only the original Kaggle documents)."""
        self.documents, self.severity = load_documents(data_dir)
        self.documents, self.enrichment = enrich_documents(
            self.documents, medlineplus=medlineplus, symptom2disease=symptom2disease)
        self.method = method
        self.symptoms = sorted({s for d in self.documents for s in d.symptoms})
        self._vsm_cache = {}
        self._boolean_cache = {}
        # surface (unstemmed) vocabulary for spelling correction and pattern queries.
        # Built from the curated text only: user descriptions contain typos, which would
        # otherwise count as "known" words and never be corrected.
        words = Counter()
        for doc in self.documents:
            words.update(w for w in tokenize(remove_punctuation(lowercase(doc.core_text)))
                         if w not in STOPWORDS and w.isalpha())
        for phrase in SYNONYMS:  # lay words such as 'tummy' are valid query words
            words.update(w for w in phrase.split() if w.isalpha())
        self.word_freq = words
        self.matcher = PatternMatcher(words, words, known_words=STOPWORDS)
        self._name_to_id = {disease_key(d.disease): d.doc_id for d in self.documents}

    def vsm(self, method=None, severity_boost=0.0):
        key = (method or self.method, severity_boost)
        if key not in self._vsm_cache:
            self._vsm_cache[key] = VectorSpaceModel(
                self.documents, key[0], self.severity, severity_boost)
        return self._vsm_cache[key]

    def boolean(self, method=None):
        method = method or self.method
        if method not in self._boolean_cache:
            self._boolean_cache[method] = BooleanModel(self.documents, method)
        return self._boolean_cache[method]

    def correct(self, text):
        """Spelling correction of a query -> (corrected text, {wrong: right})."""
        return self.matcher.correct_text(text)

    def misspellings(self, text, limit=3):
        """{word: [suggestions]} for every unknown word of the query."""
        out = {}
        for word in tokenize(remove_punctuation(lowercase(text))):
            suggestions = self.matcher.suggest(word, limit)
            if suggestions:
                out[word] = suggestions
        return out

    def suggest_symptoms(self, prefix, limit=10):
        return suggest_symptoms(prefix, self.symptoms, limit)

    def doc_id(self, disease_name):
        return self._name_to_id[disease_key(disease_name)]
