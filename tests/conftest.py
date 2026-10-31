"""Shared fixtures: a tiny hand-made collection whose results can be computed by hand."""
import pytest

from ir.data_loader import Document


@pytest.fixture
def tiny_docs():
    #  doc 0: alpha  fever rash
    #  doc 1: beta   fever cough
    #  doc 2: gamma  rash itching
    return [
        Document(0, "Alpha", "", ["fever", "rash"], []),
        Document(1, "Beta", "", ["fever", "cough"], []),
        Document(2, "Gamma", "", ["rash", "itching"], []),
    ]
