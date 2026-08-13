"""
Tests for Module 2: Text Preprocessing & Normalization.
"""

import pytest
import sys
import os

# Ensure backend package directory is on python path for importing ai module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai.preprocessing import preprocess_document, validate_input


def test_normal_academic_text():
    """TEST 1 — Normal Academic Text"""
    doc = {
        "filename": "academic.pdf",
        "source_type": "pdf",
        "text": "Machine Learning is a subset of Artificial Intelligence.\nIt allows computers to learn from data."
    }
    result = preprocess_document(doc)

    assert result["filename"] == "academic.pdf"
    assert result["source_type"] == "pdf"
    assert result["sentence_count"] == 2
    assert result["original_sentences"][0] == "Machine Learning is a subset of Artificial Intelligence."
    assert result["original_sentences"][1] == "It allows computers to learn from data."
    assert result["clean_sentences"][0] == "machine learning subset artificial intelligence"
    assert result["clean_sentences"][1] == "allows computers learn data"
    assert "machine" in result["tokens"]
    assert "learning" in result["tokens_without_stopwords"]


def test_multiple_spaces():
    """TEST 2 — Multiple Spaces"""
    doc = {
        "filename": "spaces.txt",
        "source_type": "text",
        "text": "Machine     learning     is     useful."
    }
    result = preprocess_document(doc)

    assert result["clean_sentences"][0] == "machine learning useful"
    assert result["normalized_text"] == "Machine learning is useful."


def test_newlines():
    """TEST 3 — Newlines"""
    doc = {
        "filename": "newlines.txt",
        "source_type": "text",
        "text": "Machine learning is useful.\n\nIt is widely used in AI."
    }
    result = preprocess_document(doc)

    assert result["sentence_count"] == 2
    assert len(result["original_sentences"]) == 2
    assert result["original_sentences"][0] == "Machine learning is useful."
    assert result["original_sentences"][1] == "It is widely used in AI."


def test_punctuation():
    """TEST 4 — Punctuation"""
    doc = {
        "filename": "punct.txt",
        "source_type": "text",
        "text": "Machine learning, artificial intelligence, and robotics!"
    }
    result = preprocess_document(doc)

    # Original sentence retains punctuation
    assert result["original_sentences"][0] == "Machine learning, artificial intelligence, and robotics!"
    # Clean sentence strips punctuation
    assert result["clean_sentences"][0] == "machine learning artificial intelligence robotics"


def test_empty_text_and_validation():
    """TEST 5 — Empty Text & Input Validation"""
    with pytest.raises(ValueError, match="cannot be empty"):
        preprocess_document({"filename": "empty.txt", "text": ""})

    with pytest.raises(ValueError, match="cannot be empty"):
        preprocess_document({"filename": "whitespace.txt", "text": "   \n\t  "})

    with pytest.raises(ValueError, match="Missing required 'text' field"):
        preprocess_document({"filename": "notext.txt"})

    with pytest.raises(ValueError, match="Expected a dictionary object"):
        preprocess_document("Not a dict")

    with pytest.raises(ValueError, match="'text' field must be a string"):
        preprocess_document({"text": 12345})


def test_special_unicode():
    """TEST 6 — Special Unicode"""
    doc = {
        "filename": "unicode.txt",
        "source_type": "text",
        "text": "AI — Artificial Intelligence & Machine Learning\u2019s future."
    }
    result = preprocess_document(doc)

    assert result["sentence_count"] == 1
    assert "AI" in result["original_text"]
    assert len(result["tokens"]) > 0


def test_sentence_mapping():
    """TEST 7 — Sentence Mapping Alignment"""
    doc = {
        "filename": "mapping.pdf",
        "source_type": "pdf",
        "text": "First sentence is here. Second sentence follows immediately."
    }
    result = preprocess_document(doc)

    sentences = result["sentences"]
    assert len(sentences) == 2

    # Verify sentence 0 mapping
    assert sentences[0]["index"] == 0
    assert sentences[0]["original"] == "First sentence is here."
    assert sentences[0]["clean"] == "first sentence"
    assert sentences[0]["tokens"] == ["first", "sentence"]

    # Verify sentence 1 mapping
    assert sentences[1]["index"] == 1
    assert sentences[1]["original"] == "Second sentence follows immediately."
    assert sentences[1]["clean"] == "second sentence follows immediately"
    assert sentences[1]["tokens"] == ["second", "sentence", "follows", "immediately"]


def test_dynamic_counts_and_contract_keys():
    """Verify all contract keys and dynamic stats calculation"""
    doc = {
        "filename": "contract_test.pdf",
        "source_type": "pdf",
        "text": "Deep learning models require data. Algorithms optimize parameters."
    }
    result = preprocess_document(doc)

    required_keys = [
        "filename", "source_type", "original_text", "normalized_text",
        "original_sentences", "clean_sentences", "sentences",
        "tokens", "tokens_without_stopwords",
        "sentence_count", "word_count", "character_count"
    ]
    for key in required_keys:
        assert key in result, f"Key '{key}' missing from Module 2 output contract"

    assert result["sentence_count"] == 2
    assert result["word_count"] == 8
    assert result["character_count"] == len(doc["text"])
