"""
Experiment 3 - Boolean retrieval model.

* Term-document incidence matrix: rows = terms, columns = diseases, cell = 1 if the
  term occurs in the document (pandas DataFrame, exportable to CSV).
* Query language:   fever AND rash NOT cough,   (cough OR phlegm) AND NOT "chest pain"
      - operators AND, OR, NOT (case-insensitive) and parentheses
      - adjacent words without an operator are ANDed:  "high fever" = high AND fever
      - "A NOT B" is read as  A AND NOT B
      - a quoted phrase is matched with the positional inverted index
  Precedence: NOT > AND > OR.

Grammar (recursive descent parser):
      expr   := conj ( OR conj )*
      conj   := factor ( [AND] factor | NOT factor )*
      factor := NOT factor | '(' expr ')' | WORDS | PHRASE

Evaluation works on boolean row vectors of the incidence matrix:
      AND -> element-wise &,  OR -> |,  NOT -> ~
"""

import re

import numpy as np
import pandas as pd

from ir.indexing import InvertedIndex
from ir.preprocessing import preprocess

OPERATORS = {"AND", "OR", "NOT"}
_TOKEN_RE = re.compile(r'\(|\)|"[^"]*"|[^\s()"]+')


class BooleanQueryError(ValueError):
    pass


def build_incidence_matrix(tokenized_docs, doc_names):
    """Term-document incidence matrix: M[t, d] = 1 if term t occurs in document d."""
    vocab = sorted({t for tokens in tokenized_docs for t in tokens})
    row = {t: i for i, t in enumerate(vocab)}
    matrix = np.zeros((len(vocab), len(tokenized_docs)), dtype=np.int8)
    for j, tokens in enumerate(tokenized_docs):
        for t in set(tokens):
            matrix[row[t], j] = 1
    return pd.DataFrame(matrix, index=vocab, columns=doc_names)


def tokenize_query(query):
    """Split a query into operator, parenthesis, quoted-phrase and word tokens.
    Consecutive words are grouped into one ('WORDS', text) token so that the synonym
    map can see multi-word phrases such as 'stomach ache'."""
    tokens = []
    for raw in _TOKEN_RE.findall(query):
        upper = raw.upper()
        if upper in OPERATORS:
            tokens.append((upper, raw))
        elif raw in "()":
            tokens.append((raw, raw))
        elif raw.startswith('"'):
            tokens.append(("PHRASE", raw.strip('"')))
        elif tokens and tokens[-1][0] == "WORDS":
            tokens[-1] = ("WORDS", tokens[-1][1] + " " + raw)
        else:
            tokens.append(("WORDS", raw))
    return tokens


class _Parser:
    """Recursive-descent parser producing a small AST of tuples:
        ('TERM', t) ('PHRASE', [t1, t2]) ('AND', a, b) ('OR', a, b) ('NOT', a)
    Word groups that are all stopwords become None and are ignored by the operators."""

    def __init__(self, tokens, method):
        self.tokens, self.pos, self.method = tokens, 0, method

    def peek(self):
        return self.tokens[self.pos][0] if self.pos < len(self.tokens) else None

    def take(self, kind=None):
        if self.pos >= len(self.tokens):
            raise BooleanQueryError("unexpected end of query")
        token = self.tokens[self.pos]
        if kind and token[0] != kind:
            raise BooleanQueryError(f"expected {kind} but found '{token[1]}'")
        self.pos += 1
        return token

    def parse(self):
        if not self.tokens:
            raise BooleanQueryError("empty query")
        node = self.expr()
        if self.pos != len(self.tokens):
            raise BooleanQueryError(f"unexpected '{self.tokens[self.pos][1]}'")
        return node

    def expr(self):
        node = self.conj()
        while self.peek() == "OR":
            self.take()
            node = _combine("OR", node, self.conj())
        return node

    def conj(self):
        node = self.factor()
        while self.peek() in ("AND", "NOT", "WORDS", "PHRASE", "("):
            if self.peek() == "AND":
                self.take()
                node = _combine("AND", node, self.factor())
            elif self.peek() == "NOT":       # "A NOT B" == "A AND NOT B"
                self.take()
                node = _combine("AND", node, _negate(self.factor()))
            else:                             # implicit AND
                node = _combine("AND", node, self.factor())
        return node

    def factor(self):
        kind = self.peek()
        if kind == "NOT":
            self.take()
            return _negate(self.factor())
        if kind == "(":
            self.take()
            node = self.expr()
            self.take(")")
            return node
        if kind == "WORDS":
            terms = preprocess(self.take()[1], self.method)
            node = None
            for t in terms:
                node = _combine("AND", node, ("TERM", t))
            return node
        if kind == "PHRASE":
            terms = preprocess(self.take()[1], self.method)
            return ("PHRASE", terms) if terms else None
        found = self.tokens[self.pos][1] if kind else "end of query"
        raise BooleanQueryError(f"expected a term but found '{found}'")


def _combine(op, left, right):
    if left is None:
        return right
    if right is None:
        return left
    return (op, left, right)


def _negate(node):
    return None if node is None else ("NOT", node)


def parse_query(query, method="stem"):
    """Query string -> AST (raises BooleanQueryError on syntax errors)."""
    return _Parser(tokenize_query(query), method).parse()


def ast_to_string(node):
    """Readable form of the parsed query, e.g. AND(fever, NOT(cough))."""
    if node is None:
        return "<empty>"
    kind = node[0]
    if kind == "TERM":
        return node[1]
    if kind == "PHRASE":
        return '"' + " ".join(node[1]) + '"'
    if kind == "NOT":
        return f"NOT({ast_to_string(node[1])})"
    return f"{kind}({ast_to_string(node[1])}, {ast_to_string(node[2])})"


class BooleanModel:
    """Boolean retrieval with an incidence matrix (plus a positional index for phrases)."""

    def __init__(self, documents, method="stem"):
        self.documents = documents
        self.method = method
        self.doc_tokens = [preprocess(d.text, method) for d in documents]
        self.index = InvertedIndex(self.doc_tokens)
        self.matrix = build_incidence_matrix(self.doc_tokens, [d.disease for d in documents])
        self.num_docs = len(documents)
        self.unknown_terms = []

    def export_matrix(self, path):
        self.matrix.to_csv(path)
        return path

    def term_vector(self, term):
        """Boolean row of the incidence matrix (all False for an unknown term)."""
        if term in self.matrix.index:
            return self.matrix.loc[term].to_numpy().astype(bool)
        self.unknown_terms.append(term)
        return np.zeros(self.num_docs, dtype=bool)

    def evaluate(self, node):
        if node is None:
            return np.zeros(self.num_docs, dtype=bool)
        kind = node[0]
        if kind == "TERM":
            return self.term_vector(node[1])
        if kind == "PHRASE":
            vec = np.zeros(self.num_docs, dtype=bool)
            vec[list(self.index.phrase_query(node[1]))] = True
            return vec
        if kind == "NOT":
            return ~self.evaluate(node[1])
        left, right = self.evaluate(node[1]), self.evaluate(node[2])
        return left & right if kind == "AND" else left | right

    def search(self, query):
        """Return the sorted doc ids matching the Boolean query (an unranked set)."""
        self.unknown_terms = []
        return [int(i) for i in np.flatnonzero(self.evaluate(parse_query(query, self.method)))]

    def search_terms(self, terms, operator="AND"):
        """Combine already-preprocessed terms with one operator (used for evaluation of
        plain-language queries)."""
        node = None
        for t in dict.fromkeys(terms):
            node = _combine(operator, node, ("TERM", t))
        self.unknown_terms = []
        return [int(i) for i in np.flatnonzero(self.evaluate(node))]
