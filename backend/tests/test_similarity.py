"""
Tests for Module 3: Plagiarism / Similarity Detection Engine.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai.preprocessing import preprocess_document
from ai.similarity import (
    analyze_similarity,
    load_reference_corpus,
    calculate_document_similarity,
    calculate_sentence_similarity,
    calculate_sequence_similarity
)


@pytest.fixture(scope="module")
def reference_corpus():
    """Loads the real local reference corpus from data/reference_documents/"""
    return load_reference_corpus()


def test_reference_corpus_loading(reference_corpus):
    """Ensure reference corpus loads correctly"""
    assert len(reference_corpus) > 0
    filenames = [doc["filename"] for doc in reference_corpus]
    assert "ai_fundamentals.txt" in filenames
    assert "computer_networks.txt" in filenames


def test_identical_sentence(reference_corpus):
    """Test 1: Identical sentence matching"""
    sample_text = (
        "Machine Learning is a foundational subset of Artificial Intelligence that provides "
        "systems the ability to automatically learn and improve from experience without being explicitly programmed."
    )
    preprocessed = preprocess_document({"filename": "identical.txt", "text": sample_text})
    result = analyze_similarity(preprocessed, reference_corpus)

    assert len(result["sentence_matches"]) == 1
    match = result["sentence_matches"][0]
    assert match["severity"] == "HIGH"
    assert match["similarity"] >= 90.0
    assert match["source"] == "ai_fundamentals.txt"


def test_unrelated_sentence(reference_corpus):
    """Test 2: Completely unrelated sentence"""
    sample_text = "The university cafeteria serves fresh soup during afternoon lunch hours."
    preprocessed = preprocess_document({"filename": "unrelated.txt", "text": sample_text})
    result = analyze_similarity(preprocessed, reference_corpus)

    assert len(result["sentence_matches"]) == 1
    match = result["sentence_matches"][0]
    assert match["severity"] == "LOW"
    assert match["similarity"] < 50.0


def test_multiple_matching_sources(reference_corpus):
    """Test 3: Multiple matching sources in single submission"""
    sample_text = (
        "Machine Learning is a foundational subset of Artificial Intelligence that provides "
        "systems the ability to automatically learn.\n"
        "The Open Systems Interconnection OSI reference model categorizes network architecture "
        "into seven distinct abstraction layers."
    )
    preprocessed = preprocess_document({"filename": "multi_source.txt", "text": sample_text})
    result = analyze_similarity(preprocessed, reference_corpus)

    matched_source_names = [s["source"] for s in result["matched_sources"]]
    assert "ai_fundamentals.txt" in matched_source_names
    assert "computer_networks.txt" in matched_source_names


def test_sentence_level_detection_structure(reference_corpus):
    """Test 4: Structure of sentence-level detection output"""
    sample_text = "Supervised machine learning algorithms apply what has been learned in the past."
    preprocessed = preprocess_document({"filename": "structure.txt", "text": sample_text})
    result = analyze_similarity(preprocessed, reference_corpus)

    assert "sentence_matches" in result
    assert len(result["sentence_matches"]) == 1
    m = result["sentence_matches"][0]

    assert "index" in m
    assert "submission_sentence" in m
    assert "reference_sentence" in m
    assert "source" in m
    assert "tfidf_similarity" in m
    assert "sequence_similarity" in m
    assert "similarity" in m
    assert "severity" in m


def test_sequence_matching_near_copy(reference_corpus):
    """Test 5: SequenceMatcher catches near-copy / lightly altered text"""
    # Original in dbms_normalization.txt:
    # "Database normalization is a systematic approach to decomposing relational schemas to minimize data redundancy..."
    near_copy = (
        "Database normalization is a systematic approach to decomposing relational schemas "
        "to reduce data redundancy and eliminate anomalies."
    )
    preprocessed = preprocess_document({"filename": "near_copy.txt", "text": near_copy})
    result = analyze_similarity(preprocessed, reference_corpus)

    match = result["sentence_matches"][0]
    assert match["sequence_similarity"] > 0.70
    assert match["similarity"] >= 70.0
    assert match["source"] == "dbms_normalization.txt"


def test_empty_document():
    """Test 6: Empty document handling does not crash"""
    empty_preprocessed = {
        "filename": "empty.txt",
        "source_type": "text",
        "original_text": "",
        "normalized_text": "",
        "original_sentences": [],
        "clean_sentences": [],
        "sentences": [],
        "tokens": [],
        "tokens_without_stopwords": [],
        "sentence_count": 0,
        "word_count": 0,
        "character_count": 0
    }

    result = analyze_similarity(empty_preprocessed, [])
    assert result["document_similarity"] == 0.0
    assert result["matched_sources"] == []
    assert result["sentence_matches"] == []
    assert result["statistics"]["total_sentences"] == 0


def test_missing_reference_dir():
    """Test 7: Handles non-existent reference directory gracefully"""
    corpus = load_reference_corpus("/path/does/not/exist")
    assert corpus == []


def test_qualitative_clean_vs_plagiarized_vs_paraphrased(reference_corpus):
    """Verify qualitative differentiation between clean, plagiarized, and paraphrased text"""
    # Clean text (unrelated topic)
    clean_text = (
        "Astronomy is the scientific study of celestial objects and phenomena. "
        "It applies mathematics, physics, and chemistry to explain their origin and evolution."
    )
    clean_prep = preprocess_document({"filename": "clean.txt", "text": clean_text})
    clean_res = analyze_similarity(clean_prep, reference_corpus)

    # Directly Plagiarized text (copied verbatim from ai_fundamentals.txt)
    plagiarized_text = (
        "Artificial Intelligence represents the simulation of human intelligence processes by computer systems. "
        "Machine Learning is a foundational subset of Artificial Intelligence that provides systems "
        "the ability to automatically learn and improve from experience without being explicitly programmed."
    )
    plag_prep = preprocess_document({"filename": "plag.txt", "text": plagiarized_text})
    plag_res = analyze_similarity(plag_prep, reference_corpus)

    # Paraphrased text (reworded version of climate_research.txt)
    paraphrased_text = (
        "The atmospheric greenhouse effect plays a critical role in controlling surface heat on planets. "
        "Thermal expansion of global sea waters combined with glacier melting contributes to marine habitat destruction."
    )
    para_prep = preprocess_document({"filename": "para.txt", "text": paraphrased_text})
    para_res = analyze_similarity(para_prep, reference_corpus)

    # Quantitative Assertions:
    # 1. Plagiarized similarity must be higher than clean similarity
    assert plag_res["document_similarity"] > clean_res["document_similarity"]
    assert plag_res["statistics"]["high_similarity_sentences"] >= 1

    # 2. Clean document should have low similarity
    assert clean_res["document_similarity"] < 40.0
    assert clean_res["statistics"]["high_similarity_sentences"] == 0

    # 3. Paraphrased should show moderate/meaningful similarity
    assert para_res["document_similarity"] > clean_res["document_similarity"]
