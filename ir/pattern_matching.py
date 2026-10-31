"""
Experiment 5 - Pattern matching over the index vocabulary.

Implements, from scratch:
  * exact / prefix / suffix / substring matching
  * lexical range search (all terms alphabetically between two words)
  * Levenshtein edit distance (dynamic programming) and fuzzy search
  * spelling correction ("feaver" -> "fever", "diarhea" -> "diarrhoea")
  * wildcard ("head*", "?ough") and regular-expression search
  * prefix-based symptom suggestions for the command-line demo

The vocabulary is kept sorted so that prefix and range queries can use binary search.
"""

import re


# ---------------------------------------------------------------------------
# Edit distance
# ---------------------------------------------------------------------------
def levenshtein_matrix(source, target):
    """Return the full dynamic-programming table for the edit distance.

    D[i][j] = edit distance between source[:i] and target[:j]

        D[i][0] = i                      (delete i characters)
        D[0][j] = j                      (insert j characters)
        D[i][j] = min( D[i-1][j]   + 1,                       # deletion
                       D[i][j-1]   + 1,                       # insertion
                       D[i-1][j-1] + (0 if s[i]==t[j] else 1) # substitution / match
                     )
    """
    rows, cols = len(source) + 1, len(target) + 1
    table = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        table[i][0] = i
    for j in range(cols):
        table[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if source[i - 1] == target[j - 1] else 1
            table[i][j] = min(table[i - 1][j] + 1,         # delete
                              table[i][j - 1] + 1,         # insert
                              table[i - 1][j - 1] + cost)  # substitute / match
    return table


def levenshtein(source, target):
    """Levenshtein edit distance = D[len(source)][len(target)].

    Uses only two rows of the DP table (O(min(m, n)) memory); the recurrence is the
    same as in :func:`levenshtein_matrix`.
    """
    if len(source) < len(target):
        source, target = target, source
    previous = list(range(len(target) + 1))
    for i, s_char in enumerate(source, start=1):
        current = [i] + [0] * len(target)
        for j, t_char in enumerate(target, start=1):
            cost = 0 if s_char == t_char else 1
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
        previous = current
    return previous[-1]


# ---------------------------------------------------------------------------
# Binary search helpers (vocabulary is sorted)
# ---------------------------------------------------------------------------
def lower_bound(sorted_list, key):
    """Index of the first element >= key (classic binary search)."""
    lo, hi = 0, len(sorted_list)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_list[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    return lo


def upper_bound(sorted_list, key):
    """Index of the first element > key."""
    lo, hi = 0, len(sorted_list)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_list[mid] <= key:
            lo = mid + 1
        else:
            hi = mid
    return lo


# ---------------------------------------------------------------------------
# Pattern matcher
# ---------------------------------------------------------------------------
def wildcard_to_regex(pattern):
    """Translate an extended wildcard pattern into an anchored regular expression.

        *      any sequence of characters (also empty)    head*  -> ^head.*$
        ?      exactly one character                      ?ough  -> ^.ough$
        [abc]  one character from the set (kept as is)    [cb]ough
        other characters are matched literally
    """
    regex, i = "", 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "*":
            regex += ".*"
        elif ch == "?":
            regex += "."
        elif ch == "[":
            end = pattern.find("]", i)
            if end == -1:
                regex += re.escape(ch)
            else:
                regex += pattern[i:end + 1]
                i = end
        else:
            regex += re.escape(ch)
        i += 1
    return "^" + regex + "$"


def max_edits_for(word):
    """Allowed edit distance grows with word length: <=3 chars: 1, <=8: 2, longer: 3."""
    return 1 if len(word) <= 3 else 2 if len(word) <= 8 else 3


class PatternMatcher:
    """Pattern queries over a vocabulary of (unstemmed) words.

    word_freq: optional {word: collection frequency}, used to break ties between
    equally close spelling suggestions (the more common word wins).
    known_words: extra words that are never corrected (e.g. stopwords such as 'since').
    """

    def __init__(self, vocabulary, word_freq=None, dictionary_check=True, known_words=()):
        self.vocabulary = sorted(set(vocabulary))
        self.reversed_vocab = sorted(w[::-1] for w in self.vocabulary)
        self.word_freq = word_freq or {}
        self.dictionary_check = dictionary_check
        self._vocab_set = set(self.vocabulary)
        self._known = set(known_words)

    # -- simple matches ---------------------------------------------------
    def exact_match(self, word):
        """Binary search for the word itself."""
        i = lower_bound(self.vocabulary, word)
        return [word] if i < len(self.vocabulary) and self.vocabulary[i] == word else []

    def prefix_match(self, prefix):
        """All words in [prefix, prefix + '\\uffff') - a contiguous block of the sorted list."""
        lo = lower_bound(self.vocabulary, prefix)
        hi = lower_bound(self.vocabulary, prefix + "￿")
        return self.vocabulary[lo:hi]

    def suffix_match(self, suffix):
        """Prefix search on the reversed vocabulary ('ache' -> words ending in 'ache')."""
        rev = suffix[::-1]
        lo = lower_bound(self.reversed_vocab, rev)
        hi = lower_bound(self.reversed_vocab, rev + "￿")
        return sorted(w[::-1] for w in self.reversed_vocab[lo:hi])

    def substring_match(self, sub):
        return [w for w in self.vocabulary if sub in w]

    def range_search(self, low, high):
        """Lexical range: every word w with low <= w <= high (alphabetical order)."""
        if low > high:
            low, high = high, low
        return self.vocabulary[lower_bound(self.vocabulary, low): upper_bound(self.vocabulary, high)]

    # -- pattern / regex --------------------------------------------------
    def wildcard_search(self, pattern):
        regex = re.compile(wildcard_to_regex(pattern))
        return [w for w in self.vocabulary if regex.match(w)]

    def regex_search(self, expression):
        regex = re.compile(expression)
        return [w for w in self.vocabulary if regex.search(w)]

    # -- fuzzy ------------------------------------------------------------
    def fuzzy_search(self, word, max_distance=None):
        """[(candidate, distance)] within max_distance, closest (then most frequent) first."""
        if max_distance is None:
            max_distance = max_edits_for(word)
        hits = []
        for candidate in self.vocabulary:
            if abs(len(candidate) - len(word)) > max_distance:  # cheap length filter
                continue
            d = levenshtein(word, candidate)
            if d <= max_distance:
                hits.append((candidate, d))
        hits.sort(key=lambda x: (x[1], -self.word_freq.get(x[0], 0), x[0]))
        return hits

    def is_known(self, word):
        """A word is not corrected if it is in the vocabulary or is a valid English word
        (WordNet), e.g. 'night' or 'since' must not become a symptom word."""
        if word in self._vocab_set or word in self._known or len(word) <= 2 or word.isdigit():
            return True
        if self.dictionary_check:
            try:
                from nltk.corpus import wordnet
                return bool(wordnet.synsets(word))
            except LookupError:
                return False
        return False

    def suggest(self, word, limit=3):
        """'Did you mean ...?' candidates for an unknown word ([] if the word is known)."""
        if self.is_known(word):
            return []
        return [w for w, _ in self.fuzzy_search(word)[:limit]]

    def correct_word(self, word):
        """Best correction or the word itself."""
        suggestions = self.suggest(word, limit=1)
        return suggestions[0] if suggestions else word

    def correct_text(self, text):
        """Correct every word of a query.  Returns (corrected_text, {wrong: right})."""
        corrections = {}

        def fix(match):
            word = match.group(0)
            fixed = self.correct_word(word.lower())
            if fixed != word.lower():
                corrections[word] = fixed
                return fixed
            return word

        return re.sub(r"[A-Za-z]+", fix, text), corrections


def suggest_symptoms(prefix, symptoms, limit=10):
    """Symptom names starting with `prefix`, or having a word that starts with it
    ('he' -> 'headache', 'high fever', ...).  Used for auto-suggestions in the demo."""
    prefix = prefix.lower().strip()
    starts = [s for s in symptoms if s.startswith(prefix)]
    inner = [s for s in symptoms if s not in starts
             and any(w.startswith(prefix) for w in s.split())]
    return (sorted(starts) + sorted(inner))[:limit]
