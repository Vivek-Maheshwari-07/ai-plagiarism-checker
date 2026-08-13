import json
import time
import pytest
from pathlib import Path
from ai.scoring import (
    calculate_score,
    calculate_source_breadth,
    classify_risk,
    DOCUMENT_WEIGHT,
    FLAGGED_SENTENCE_WEIGHT,
    SOURCE_BREADTH_WEIGHT
)


def load_fixture(name: str) -> dict:
    fixture_path = Path(__file__).parent / "fixtures" / name
    with open(fixture_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_1_no_similarity():
    """TEST 1 — No similarity -> final_score = 0, risk = LOW"""
    input_data = {
        "document_similarity": 0.0,
        "matched_sources": [],
        "sentence_matches": [],
        "statistics": {
            "total_sentences": 20,
            "high_similarity_sentences": 0,
            "moderate_similarity_sentences": 0,
            "flagged_sentence_percentage": 0.0,
            "sources_matched": 0
        }
    }
    result = calculate_score(input_data)
    assert result["final_score"] == 0.0
    assert result["risk_level"] == "LOW"
    assert result["evidence_strength"] == "LOW"
    assert result["score_breakdown"]["source_breadth"]["value"] == 0.0


def test_2_low_similarity():
    """TEST 2 — Low similarity -> risk = LOW"""
    # doc_sim = 20 (10.0), flagged = 10 (3.0), sources = 1 (40 * 0.20 = 8.0) -> total = 21.0
    input_data = {
        "document_similarity": 20.0,
        "matched_sources": [{"source": "ref1.txt", "similarity": 20.0}],
        "sentence_matches": [],
        "statistics": {
            "total_sentences": 20,
            "high_similarity_sentences": 0,
            "moderate_similarity_sentences": 1,
            "flagged_sentence_percentage": 10.0,
            "sources_matched": 1
        }
    }
    result = calculate_score(input_data)
    assert result["final_score"] == 21.0
    assert result["risk_level"] == "LOW"


def test_3_moderate_result():
    """TEST 3 — Moderate result -> risk = MODERATE (between 25.1 and 50)"""
    # doc_sim = 50 (25.0), flagged = 20 (6.0), sources = 1 (8.0) -> total = 39.0
    input_data = {
        "document_similarity": 50.0,
        "matched_sources": [{"source": "ref1.txt", "similarity": 50.0}],
        "sentence_matches": [],
        "statistics": {
            "total_sentences": 20,
            "high_similarity_sentences": 2,
            "moderate_similarity_sentences": 2,
            "flagged_sentence_percentage": 20.0,
            "sources_matched": 1
        }
    }
    result = calculate_score(input_data)
    assert result["final_score"] == 39.0
    assert result["risk_level"] == "MODERATE"


def test_4_high_result():
    """TEST 4 — High result -> risk = HIGH (between 50.1 and 75)"""
    # doc_sim = 70 (35.0), flagged = 50 (15.0), sources = 2 (14.0) -> total = 64.0
    input_data = {
        "document_similarity": 70.0,
        "matched_sources": [
            {"source": "ref1.txt", "similarity": 70.0},
            {"source": "ref2.txt", "similarity": 60.0}
        ],
        "sentence_matches": [],
        "statistics": {
            "total_sentences": 20,
            "high_similarity_sentences": 5,
            "moderate_similarity_sentences": 5,
            "flagged_sentence_percentage": 50.0,
            "sources_matched": 2
        }
    }
    result = calculate_score(input_data)
    assert result["final_score"] == 64.0
    assert result["risk_level"] == "HIGH"


def test_5_critical_result():
    """TEST 5 — Critical result -> risk = CRITICAL (>75.0)"""
    # doc_sim = 95 (47.5), flagged = 80 (24.0), sources = 3 (20.0) -> total = 91.5
    input_data = {
        "document_similarity": 95.0,
        "matched_sources": [
            {"source": "ref1.txt", "similarity": 95.0},
            {"source": "ref2.txt", "similarity": 80.0},
            {"source": "ref3.txt", "similarity": 75.0}
        ],
        "sentence_matches": [
            {"severity": "HIGH", "similarity": 95.0},
            {"severity": "HIGH", "similarity": 90.0}
        ],
        "statistics": {
            "total_sentences": 20,
            "high_similarity_sentences": 12,
            "moderate_similarity_sentences": 4,
            "flagged_sentence_percentage": 80.0,
            "sources_matched": 3
        }
    }
    result = calculate_score(input_data)
    assert result["final_score"] == 91.5
    assert result["risk_level"] == "CRITICAL"
    assert result["evidence_strength"] == "HIGH"


def test_6_boundary_25():
    """TEST 6 — Boundary exactly 25.0 -> LOW"""
    assert classify_risk(25.0) == "LOW"
    assert classify_risk(25.0000) == "LOW"
    assert classify_risk(25.1) == "MODERATE"


def test_7_boundary_50():
    """TEST 7 — Boundary exactly 50.0 -> MODERATE"""
    assert classify_risk(50.0) == "MODERATE"
    assert classify_risk(50.1) == "HIGH"


def test_8_boundary_75():
    """TEST 8 — Boundary exactly 75.0 -> HIGH"""
    assert classify_risk(75.0) == "HIGH"
    assert classify_risk(75.1) == "CRITICAL"


def test_9_multiple_sources_normalization():
    """TEST 9 — Multiple Sources normalization rule"""
    assert calculate_source_breadth(0) == 0.0
    assert calculate_source_breadth(1) == 40.0
    assert calculate_source_breadth(2) == 70.0
    assert calculate_source_breadth(3) == 100.0
    assert calculate_source_breadth(5) == 100.0


def test_10_zero_sentences_validation():
    """TEST 10 — Zero sentences validation -> raises ValueError"""
    input_data = {
        "document_similarity": 0.0,
        "matched_sources": [],
        "sentence_matches": [],
        "statistics": {
            "total_sentences": 0,
            "sources_matched": 0
        }
    }
    with pytest.raises(ValueError, match="no analyzable sentences"):
        calculate_score(input_data)


def test_11_empty_sources():
    """TEST 11 — Empty sources -> source breadth = 0, no crash"""
    input_data = {
        "document_similarity": 40.0,
        "matched_sources": [],
        "sentence_matches": [],
        "statistics": {
            "total_sentences": 10,
            "high_similarity_sentences": 0,
            "moderate_similarity_sentences": 0,
            "flagged_sentence_percentage": 0.0,
            "sources_matched": 0
        }
    }
    result = calculate_score(input_data)
    assert result["score_breakdown"]["source_breadth"]["value"] == 0.0
    assert result["top_sources"] == []
    # 40 * 0.5 = 20.0
    assert result["final_score"] == 20.0


def test_12_score_clamping():
    """TEST 12 — Score Clamping (<0 or >100 values)"""
    input_data = {
        "document_similarity": 110.0,  # Clamped to 100
        "matched_sources": [{"source": "ref.txt", "similarity": 100.0}],
        "sentence_matches": [],
        "statistics": {
            "total_sentences": 10,
            "flagged_sentence_percentage": -10.0,  # Clamped to 0
            "sources_matched": 1
        }
    }
    result = calculate_score(input_data)
    # doc (100 * 0.5 = 50) + flagged (0 * 0.3 = 0) + source (40 * 0.2 = 8) = 58.0
    assert result["final_score"] == 58.0
    assert result["risk_level"] == "HIGH"


def test_13_determinism():
    """TEST 13 — Determinism across multiple executions"""
    input_data = load_fixture("sample_detection_result.json")
    results = [calculate_score(input_data) for _ in range(10)]
    
    first_result = results[0]
    for r in results[1:]:
        assert r == first_result


def test_14_standard_example_verification():
    """TEST 14 — Verify standard formula calculation example from spec"""
    # Example: doc_sim = 91.2, flagged_pct = 40.0, sources_matched = 2 (70.0)
    # doc = 91.2 * 0.50 = 45.6
    # sent = 40.0 * 0.30 = 12.0
    # source = 70.0 * 0.20 = 14.0
    # total = 45.6 + 12.0 + 14.0 = 71.6 -> HIGH
    input_data = load_fixture("sample_detection_result.json")
    result = calculate_score(input_data)

    assert result["final_score"] == 71.6
    assert result["risk_level"] == "HIGH"
    assert result["evidence_strength"] == "HIGH"
    assert result["score_breakdown"]["document_similarity"]["contribution"] == 45.6
    assert result["score_breakdown"]["flagged_sentence_percentage"]["contribution"] == 12.0
    assert result["score_breakdown"]["source_breadth"]["contribution"] == 14.0


def test_15_performance():
    """TEST 15 — Performance test: execution speed < 50ms"""
    input_data = load_fixture("sample_detection_result.json")
    start = time.perf_counter()

    for _ in range(100):
        calculate_score(input_data)

    elapsed_ms = (time.perf_counter() - start) * 1000
    avg_ms = elapsed_ms / 100

    assert avg_ms < 50.0, f"Average scoring execution time too slow: {avg_ms:.2f} ms"
