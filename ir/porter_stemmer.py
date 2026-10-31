"""
Porter stemmer (M.F. Porter, 1980, "An algorithm for suffix stripping"), written from scratch.

Key definitions
---------------
* A consonant is a letter other than a, e, i, o, u, and other than 'y' preceded by a
  consonant.  Every word can be written as  [C](VC)^m[V]  where C / V are runs of
  consonants / vowels.  m is the *measure* of the word (or stem).
      m=0: tree, by        m=1: trouble, oats       m=2: troubles, private
* Conditions used by the rules:
      *v*  the stem contains a vowel
      *d   the stem ends with a double consonant (-tt, -ss ...)
      *o   the stem ends consonant-vowel-consonant, last consonant not w, x or y (-hop)
      *S   the stem ends with S (similarly *L, *T, *Z)

Within one step only the LONGEST matching suffix is considered; if its condition fails
the step leaves the word unchanged (the rule lists below are ordered so that the first
suffix that matches is the longest one).
"""

VOWELS = set("aeiou")


def _is_consonant(word, i):
    """True if word[i] is a consonant ('y' is a consonant only after a vowel or at start)."""
    ch = word[i]
    if ch in VOWELS:
        return False
    if ch == "y":
        return i == 0 or not _is_consonant(word, i - 1)
    return True


def measure(stem):
    """m in [C](VC)^m[V]: the number of vowel-run -> consonant-run transitions."""
    pattern = "".join("c" if _is_consonant(stem, i) else "v" for i in range(len(stem)))
    return pattern.count("vc")


def _contains_vowel(stem):
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def _ends_double_consonant(stem):
    return len(stem) >= 2 and stem[-1] == stem[-2] and _is_consonant(stem, len(stem) - 1)


def _ends_cvc(stem):
    """*o condition: consonant-vowel-consonant ending, last letter not w, x or y."""
    return (len(stem) >= 3
            and _is_consonant(stem, len(stem) - 3)
            and not _is_consonant(stem, len(stem) - 2)
            and _is_consonant(stem, len(stem) - 1)
            and stem[-1] not in "wxy")


def _m_gt0(stem):
    return measure(stem) > 0


def _m_gt1(stem):
    return measure(stem) > 1


def _apply_rules(word, rules):
    """Apply the first (= longest) rule whose suffix matches; stop even if its condition fails."""
    for suffix, replacement, condition in rules:
        if word.endswith(suffix):
            stem = word[: len(word) - len(suffix)]
            if condition is None or condition(stem):
                return stem + replacement
            return word
    return word


def step1a(word):
    """SSES->SS, IES->I, SS->SS, S->''   (caresses->caress, ponies->poni, cats->cat)."""
    return _apply_rules(word, [("sses", "ss", None), ("ies", "i", None),
                               ("ss", "ss", None), ("s", "", None)])


def step1b(word):
    """(m>0) EED->EE; (*v*) ED->''; (*v*) ING->''  then tidy-up (hopping->hop, filing->file)."""
    if word.endswith("eed"):
        stem = word[:-3]
        return stem + "ee" if measure(stem) > 0 else word
    for suffix in ("ed", "ing"):
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            if not _contains_vowel(stem):
                return word
            # second or third rule succeeded -> tidy-up rules
            if stem.endswith(("at", "bl", "iz")):
                return stem + "e"                         # conflat(ed)->conflate
            if _ends_double_consonant(stem) and stem[-1] not in "lsz":
                return stem[:-1]                          # hopp(ing)->hop
            if measure(stem) == 1 and _ends_cvc(stem):
                return stem + "e"                         # fil(ing)->file
            return stem
    return word


def step1c(word):
    """(*v*) Y->I   (happy->happi, sky->sky)."""
    return _apply_rules(word, [("y", "i", _contains_vowel)])


STEP2_RULES = [(s, r, _m_gt0) for s, r in [
    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
    ("izer", "ize"), ("abli", "able"), ("alli", "al"), ("entli", "ent"), ("eli", "e"),
    ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"), ("ator", "ate"),
    ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"), ("ousness", "ous"),
    ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
]]

STEP3_RULES = [(s, r, _m_gt0) for s, r in [
    ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
    ("ical", "ic"), ("ful", ""), ("ness", ""),
]]

STEP4_RULES = [(s, "", _m_gt1) for s in [
    "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent",
]] + [("ion", "", lambda stem: measure(stem) > 1 and stem.endswith(("s", "t")))] + [
    (s, "", _m_gt1) for s in ["ou", "ism", "ate", "iti", "ous", "ive", "ize"]
]


def step2(word):
    """(m>0) double suffixes -> single ones (relational->relate, sensitiviti->sensitive)."""
    return _apply_rules(word, STEP2_RULES)


def step3(word):
    """(m>0) -ic-, -full, -ness etc. (hopeful->hope, goodness->good)."""
    return _apply_rules(word, STEP3_RULES)


def step4(word):
    """(m>1) remove -ant, -ence, -ment ... (revival->reviv, adjustment->adjust)."""
    return _apply_rules(word, STEP4_RULES)


def step5a(word):
    """(m>1) E->'';  (m=1 and not *o) E->''   (probate->probat, rate->rate)."""
    if word.endswith("e"):
        stem = word[:-1]
        m = measure(stem)
        if m > 1 or (m == 1 and not _ends_cvc(stem)):
            return stem
    return word


def step5b(word):
    """(m>1 and *d and *L) -> single letter   (controll->control, roll->roll)."""
    if word.endswith("ll") and measure(word[:-1]) > 1:
        return word[:-1]
    return word


class PorterStemmer:
    """Callable stemmer with a cache (the same word is often stemmed many times)."""

    def __init__(self):
        self._cache = {}

    def stem(self, word):
        word = word.lower()
        if word in self._cache:
            return self._cache[word]
        result = word
        for step in (step1a, step1b, step1c, step2, step3, step4, step5a, step5b):
            result = step(result)
        self._cache[word] = result
        return result

    __call__ = stem
