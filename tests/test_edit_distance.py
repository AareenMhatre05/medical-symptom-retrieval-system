import pytest

from ir.pattern_matching import PatternMatcher, levenshtein, levenshtein_matrix


@pytest.mark.parametrize("a, b, expected", [
    ("kitten", "sitting", 3),   # k->s, e->i, +g
    ("flaw", "lawn", 2),        # -f, +n
    ("", "abc", 3),
    ("abc", "", 3),
    ("fever", "fever", 0),
    ("feaver", "fever", 1),     # delete 'a'
    ("diarhea", "diarrhea", 1),
    ("diarhea", "diarrhoea", 2),
    ("intention", "execution", 5),
])
def test_levenshtein_known_values(a, b, expected):
    assert levenshtein(a, b) == expected


def test_levenshtein_symmetric():
    assert levenshtein("headake", "headache") == levenshtein("headache", "headake") == 2


def test_matrix_matches_two_row_version_and_borders():
    table = levenshtein_matrix("cat", "cut")
    #       ""  c  u  t
    #  ""  [0,  1, 2, 3]
    #  c   [1,  0, 1, 2]
    #  a   [2,  1, 1, 2]
    #  t   [3,  2, 2, 1]
    assert table == [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 1, 2], [3, 2, 2, 1]]
    assert table[-1][-1] == levenshtein("cat", "cut")


def test_spelling_correction_uses_closest_word():
    pm = PatternMatcher(["fever", "diarrhoea", "headache", "cough"], dictionary_check=False)
    assert pm.correct_word("feaver") == "fever"
    assert pm.correct_word("diarhea") == "diarrhoea"
    assert pm.correct_word("headake") == "headache"
    assert pm.correct_word("fever") == "fever"          # known words are untouched
    assert pm.suggest("zzzzzzzz") == []                 # nothing within the threshold


def test_pattern_queries():
    pm = PatternMatcher(["head", "headache", "backache", "fever", "fatigue", "cough"],
                        dictionary_check=False)
    assert pm.prefix_match("head") == ["head", "headache"]
    assert pm.suffix_match("ache") == ["backache", "headache"]
    assert pm.range_search("f", "fz") == ["fatigue", "fever"]
    assert pm.wildcard_search("head*") == ["head", "headache"]
    assert pm.wildcard_search("?ough") == ["cough"]
    assert pm.regex_search(r"^f.*e$") == ["fatigue"]
    assert pm.exact_match("fever") == ["fever"] and pm.exact_match("feve") == []
