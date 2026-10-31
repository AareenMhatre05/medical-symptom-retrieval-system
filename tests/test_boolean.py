import pytest

from ir.boolean_model import (BooleanModel, BooleanQueryError, ast_to_string, parse_query,
                              tokenize_query)


def parsed(query):
    return ast_to_string(parse_query(query, method="none"))


def test_tokenizer_groups_words_and_recognises_operators():
    assert tokenize_query('fever and (rash OR "chest pain")') == [
        ("WORDS", "fever"), ("AND", "and"), ("(", "("), ("WORDS", "rash"),
        ("OR", "OR"), ("PHRASE", "chest pain"), (")", ")")]


def test_precedence_not_and_or():
    assert parsed("fever OR rash AND cough") == "OR(fever, AND(rash, cough))"
    assert parsed("NOT fever AND rash") == "AND(NOT(fever), rash)"
    assert parsed("(fever OR rash) AND cough") == "AND(OR(fever, rash), cough)"


def test_binary_not_and_implicit_and():
    assert parsed("fever AND rash NOT cough") == "AND(AND(fever, rash), NOT(cough))"
    assert parsed("fever rash") == "AND(fever, rash)"


@pytest.mark.parametrize("bad", ["fever AND", "(fever OR rash", "fever )", "AND fever", "", "NOT"])
def test_syntax_errors(bad):
    with pytest.raises(BooleanQueryError):
        parse_query(bad, method="none")


def test_evaluation_on_tiny_collection(tiny_docs):
    model = BooleanModel(tiny_docs, method="none")
    assert model.search("fever AND rash") == [0]
    assert model.search("fever OR rash") == [0, 1, 2]
    assert model.search("fever NOT cough") == [0]
    assert model.search("NOT fever") == [2]
    assert model.search("(fever OR itching) AND NOT rash") == [1]
    assert model.search("rash itching") == [2]
    assert model.search('"fever rash"') == [0]      # positional phrase match
    assert model.search('"rash fever"') == []
    assert model.search("toothache") == [] and model.unknown_terms == ["toothache"]


def test_incidence_matrix(tiny_docs):
    m = BooleanModel(tiny_docs, method="none").matrix
    assert m.loc["fever"].tolist() == [1, 1, 0]
    assert m.loc["itching"].tolist() == [0, 0, 1]
    assert m.shape == (7, 3)   # alpha beta gamma fever rash cough itching


def test_search_terms_operator(tiny_docs):
    model = BooleanModel(tiny_docs, method="none")
    assert model.search_terms(["fever", "rash"], "AND") == [0]
    assert model.search_terms(["cough", "itching"], "OR") == [1, 2]
