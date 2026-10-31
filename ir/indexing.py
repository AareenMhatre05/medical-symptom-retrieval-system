"""
Experiment 7 - Index structures, all built from scratch.

1. InvertedIndex      term -> posting list [(doc_id, tf, [positions])]
                      Used by the vector space model to fetch candidate documents
                      (only documents sharing at least one term with the query are scored)
                      and supports positional phrase queries.
2. SuffixArray        sorted array of all suffixes of the concatenated vocabulary
                      "abdomin$acid$...$"; substring lookup by binary search.
3. SuffixTrie         generalised suffix trie (uncompressed suffix tree) of every
                      vocabulary term; substring lookup by walking down from the root.

compare_indexes() answers the same substring queries with all three structures and
reports the results and the lookup time of each.
"""

import time
from collections import defaultdict

import pandas as pd


# ---------------------------------------------------------------------------
# 1. Inverted index
# ---------------------------------------------------------------------------
class Posting:
    """One entry of a posting list."""
    __slots__ = ("doc_id", "tf", "positions")

    def __init__(self, doc_id):
        self.doc_id = doc_id
        self.tf = 0
        self.positions = []

    def __repr__(self):
        return f"(doc={self.doc_id}, tf={self.tf}, pos={self.positions})"


class InvertedIndex:
    """term -> list of Posting objects sorted by doc_id.

    df(t) = length of the posting list of t;  tf(t, d) = Posting.tf
    """

    def __init__(self, tokenized_docs):
        """tokenized_docs: list of token lists, the list index is the doc_id."""
        self.index = defaultdict(list)
        self.doc_lengths = []
        for doc_id, tokens in enumerate(tokenized_docs):
            self.doc_lengths.append(len(tokens))
            current = {}
            for position, term in enumerate(tokens):
                posting = current.get(term)
                if posting is None:
                    posting = current[term] = Posting(doc_id)
                    self.index[term].append(posting)
                posting.tf += 1
                posting.positions.append(position)
        self.index = dict(self.index)
        self.num_docs = len(tokenized_docs)

    @property
    def vocabulary(self):
        return sorted(self.index)

    def postings(self, term):
        return self.index.get(term, [])

    def document_frequency(self, term):
        return len(self.index.get(term, []))

    def docs_for_term(self, term):
        return {p.doc_id for p in self.postings(term)}

    def candidates(self, terms):
        """Union of the posting lists of the query terms."""
        result = set()
        for term in terms:
            result |= self.docs_for_term(term)
        return result

    def phrase_query(self, terms):
        """Documents containing the terms at consecutive positions (positional index)."""
        if not terms:
            return set()
        docs = set.intersection(*(self.docs_for_term(t) for t in terms))
        result = set()
        for doc_id in docs:
            positions = [next(p.positions for p in self.postings(t) if p.doc_id == doc_id)
                         for t in terms]
            first = set(positions[0])
            if any(all(start + k in set(positions[k]) for k in range(1, len(terms)))
                   for start in first):
                result.add(doc_id)
        return result

    def substring_terms(self, pattern):
        """Baseline substring lookup: linear scan over the vocabulary."""
        return {term for term in self.index if pattern in term}

    def to_dataframe(self):
        rows = [{"term": t, "df": len(p), "postings": " ".join(
            f"{x.doc_id}:{x.tf}:{x.positions}" for x in p)} for t, p in sorted(self.index.items())]
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Suffix array
# ---------------------------------------------------------------------------
SEPARATOR = "$"


def build_suffix_array(text):
    """Prefix-doubling (Manber & Myers) construction, O(n log^2 n).

    rank_k[i] = rank of the first 2^k characters of suffix i.  Suffixes are sorted by
    the pair (rank_k[i], rank_k[i + 2^k]); after round k ranks describe 2^(k+1)
    characters.  Stop when all ranks are distinct.
    """
    n = len(text)
    if n == 0:
        return []
    rank = [ord(c) for c in text]
    sa = list(range(n))
    k = 1
    while True:
        key = lambda i: (rank[i], rank[i + k] if i + k < n else -1)
        sa.sort(key=key)
        new_rank = [0] * n
        for idx in range(1, n):
            new_rank[sa[idx]] = new_rank[sa[idx - 1]] + (key(sa[idx]) != key(sa[idx - 1]))
        rank = new_rank
        if rank[sa[-1]] == n - 1:
            break
        k *= 2
    return sa


class SuffixArray:
    """Suffix array over  term_0 $ term_1 $ ... $ term_{V-1} $ ."""

    def __init__(self, vocabulary):
        self.terms = sorted(vocabulary)
        self.text = SEPARATOR.join(self.terms) + SEPARATOR
        self.starts = []  # start offset of every term in self.text
        offset = 0
        for term in self.terms:
            self.starts.append(offset)
            offset += len(term) + 1
        self.sa = build_suffix_array(self.text)

    def _term_at(self, position):
        """Which term contains text position `position` (binary search over starts)."""
        lo, hi = 0, len(self.starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.starts[mid] <= position:
                lo = mid
            else:
                hi = mid - 1
        return self.terms[lo]

    def _bound(self, pattern, upper):
        """First index in sa whose suffix prefix is >= pattern (or > pattern if upper)."""
        lo, hi, m = 0, len(self.sa), len(pattern)
        while lo < hi:
            mid = (lo + hi) // 2
            prefix = self.text[self.sa[mid]: self.sa[mid] + m]
            if prefix < pattern or (upper and prefix == pattern):
                lo = mid + 1
            else:
                hi = mid
        return lo

    def search(self, pattern):
        """All vocabulary terms that contain `pattern`.

        Suffixes starting with the pattern form one contiguous block sa[lo:hi]; two
        binary searches (O(m log n)) find its boundaries.
        """
        if not pattern or SEPARATOR in pattern:
            return set()
        lo, hi = self._bound(pattern, False), self._bound(pattern, True)
        return {self._term_at(self.sa[i]) for i in range(lo, hi)}

    def prefix_search(self, prefix):
        """Terms that START with prefix: the match must begin at a term start."""
        start_set = set(self.starts)
        lo, hi = self._bound(prefix, False), self._bound(prefix, True)
        return {self._term_at(self.sa[i]) for i in range(lo, hi) if self.sa[i] in start_set}


# ---------------------------------------------------------------------------
# 3. Generalised suffix trie
# ---------------------------------------------------------------------------
class TrieNode:
    __slots__ = ("children", "terms")

    def __init__(self):
        self.children = {}
        self.terms = set()  # ids of the vocabulary terms that pass through this node


class SuffixTrie:
    """Generalised suffix trie: every suffix of every term is inserted.

    A substring s of a term t is a prefix of some suffix of t, so walking the trie
    along s reaches a node whose `terms` set lists every term containing s.
    Lookup cost O(m) for a pattern of length m.
    """

    def __init__(self, vocabulary):
        self.terms = sorted(vocabulary)
        self.root = TrieNode()
        self.num_nodes = 1
        for term_id, term in enumerate(self.terms):
            for start in range(len(term)):
                node = self.root
                for ch in term[start:]:
                    child = node.children.get(ch)
                    if child is None:
                        child = node.children[ch] = TrieNode()
                        self.num_nodes += 1
                    node = child
                    node.terms.add(term_id)

    def search(self, pattern):
        node = self.root
        for ch in pattern:
            node = node.children.get(ch)
            if node is None:
                return set()
        return {self.terms[i] for i in node.terms}


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def _time_lookup(fn, pattern, repeat):
    start = time.perf_counter()
    for _ in range(repeat):
        result = fn(pattern)
    elapsed = (time.perf_counter() - start) / repeat
    return result, elapsed * 1e6  # microseconds


def compare_indexes(index, patterns, repeat=200):
    """Answer the same substring queries with the three structures.

    Returns (results DataFrame, build-statistics dict).
    """
    vocab = index.vocabulary
    t0 = time.perf_counter()
    sarray = SuffixArray(vocab)
    t1 = time.perf_counter()
    strie = SuffixTrie(vocab)
    t2 = time.perf_counter()
    stats = {
        "vocabulary_size": len(vocab),
        "suffix_array_length": len(sarray.sa),
        "suffix_array_build_ms": (t1 - t0) * 1e3,
        "suffix_trie_nodes": strie.num_nodes,
        "suffix_trie_build_ms": (t2 - t1) * 1e3,
    }
    rows = []
    for pattern in patterns:
        r_inv, t_inv = _time_lookup(index.substring_terms, pattern, repeat)
        r_sa, t_sa = _time_lookup(sarray.search, pattern, repeat)
        r_st, t_st = _time_lookup(strie.search, pattern, repeat)
        docs = index.candidates(r_inv)
        rows.append({
            "pattern": pattern,
            "matching_terms": ", ".join(sorted(r_inv)) if len(r_inv) <= 6
            else ", ".join(sorted(r_inv)[:6]) + f" ... (+{len(r_inv) - 6})",
            "num_terms": len(r_inv),
            "num_docs": len(docs),
            "same_result": r_inv == r_sa == r_st,
            "inverted_scan_us": round(t_inv, 2),
            "suffix_array_us": round(t_sa, 2),
            "suffix_trie_us": round(t_st, 2),
        })
    return pd.DataFrame(rows), stats
