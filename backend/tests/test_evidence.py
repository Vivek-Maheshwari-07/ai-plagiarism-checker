import json
import time
import pytest
from pathlib import Path
from ai.evidence import build_evidence, select_top_matches, group_matches_by_source


def load_fixture(name: str) -> dict:
    fixture_path = Path(__file__).parent / "fixtures" / name
    with open(fixture_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_1_strong_plagiarism_evidence():
    """TEST 1 — Strong plagiarism evidence (score 80, CRITICAL risk, 3 sources)"""
    detection_result = load_fixture("sample_detection_result.json")
    # Mock scoring result for score 71.6 HIGH
    scoring_result = {
        "final_score": 71.6,
        "risk_level": "HIGH",
        "evidence_strength": "HIGH",
        "score_breakdown": {
            "document_similarity": {"value": 91.2, "weight": 0.5, "contribution": 45.6},
            "flagged_sentence_percentage": {"value": 40.0, "weight": 0.3, "contribution": 12.0},
            "source_breadth": {"value": 70.0, "weight": 0.2, "contribution": 14.0}
        },
        "statistics": {
            "total_sentences": 20,
            "high_similarity_sentences": 8,
            "moderate_similarity_sentences": 3,
            "flagged_sentence_percentage": 40.0,
            "sources_matched": 2
        },
        "top_sources": [
            {"source": "ai_fundamentals.txt", "similarity": 91.2},
            {"source": "computer_networks.txt", "similarity": 74.3}
        ]
    }

    result = build_evidence(detection_result, scoring_result)

    assert result["score"] == 71.6
    assert result["risk_level"] == "HIGH"
    assert result["evidence_strength"] == "HIGH"
    assert len(result["top_matches"]) > 0
    assert len(result["sources"]) == 2
    assert len(result["why_flagged"]) >= 3
    assert "analysis_context" in result
    assert result["analysis_context"]["score"] == 71.6


def test_2_low_similarity_clean_document():
    """TEST 2 — Low similarity clean document -> No significant overlap explanation"""
    detection_result = {
        "document_similarity": 5.0,
        "matched_sources": [],
        "sentence_matches": [],
        "statistics": {
            "total_sentences": 15,
            "high_similarity_sentences": 0,
            "moderate_similarity_sentences": 0,
            "flagged_sentence_percentage": 0.0,
            "sources_matched": 0
        }
    }
    scoring_result = {
        "final_score": 2.5,
        "risk_level": "LOW",
        "evidence_strength": "LOW",
        "statistics": {
            "total_sentences": 15,
            "high_similarity_sentences": 0,
            "moderate_similarity_sentences": 0,
            "flagged_sentence_percentage": 0.0,
            "sources_matched": 0
        },
        "top_sources": []
    }

    result = build_evidence(detection_result, scoring_result)

    assert result["score"] == 2.5
    assert result["risk_level"] == "LOW"
    assert result["top_matches"] == []
    assert result["sources"] == []
    assert "No significant textual overlap was detected in the reference corpus." in result["why_flagged"][0]


def test_3_multiple_sources():
    """TEST 3 — Multiple sources grouping & metrics"""
    matched_sources = [
        {"source": "sourceA.txt", "similarity": 90.0},
        {"source": "sourceB.txt", "similarity": 80.0},
        {"source": "sourceC.txt", "similarity": 70.0}
    ]
    sentence_matches = [
        {"index": 0, "submission_sentence": "Text A", "reference_sentence": "Text A ref", "source": "sourceA.txt", "similarity": 90.0, "severity": "HIGH"},
        {"index": 1, "submission_sentence": "Text B", "reference_sentence": "Text B ref", "source": "sourceB.txt", "similarity": 80.0, "severity": "HIGH"},
        {"index": 2, "submission_sentence": "Text C", "reference_sentence": "Text C ref", "source": "sourceC.txt", "similarity": 70.0, "severity": "MODERATE"}
    ]
    grouped = group_matches_by_source(matched_sources, sentence_matches)

    assert len(grouped) == 3
    source_names = [s["source"] for s in grouped]
    assert "sourceA.txt" in source_names
    assert "sourceB.txt" in source_names
    assert "sourceC.txt" in source_names


def test_4_top_5_evidence_capping():
    """TEST 4 — More than 5 flagged sentences are capped at 5 in top_matches while preserving total count"""
    sentence_matches = [
        {"index": i, "submission_sentence": f"Sentence {i}", "reference_sentence": f"Ref {i}", "source": "ref.txt", "similarity": 90.0 - i, "severity": "HIGH"}
        for i in range(10)
    ]
    top_matches, all_flagged, highest_sim = select_top_matches(sentence_matches)

    assert len(top_matches) == 5
    assert len(all_flagged) == 10
    assert top_matches[0]["similarity"] == 90.0
    assert top_matches[4]["similarity"] == 86.0


def test_5_sentence_indexing_and_association():
    """TEST 5 — Sentence index and metadata integrity preserved"""
    sentence_matches = [
        {
            "index": 7,
            "submission_sentence": "Machine learning algorithms optimize parameters.",
            "reference_sentence": "Machine learning algorithms optimize model parameters.",
            "source": "ai_paper.txt",
            "similarity": 92.5,
            "tfidf_similarity": 0.91,
            "sequence_similarity": 0.94,
            "severity": "HIGH"
        }
    ]
    top_matches, _, _ = select_top_matches(sentence_matches)
    match = top_matches[0]

    assert match["index"] == 7
    assert match["submitted_text"] == "Machine learning algorithms optimize parameters."
    assert match["matched_text"] == "Machine learning algorithms optimize model parameters."
    assert match["source"] == "ai_paper.txt"
    assert match["similarity"] == 92.5
    assert match["severity"] == "HIGH"


def test_6_duplicate_source_grouping():
    """TEST 6 — Multiple sentences from one source are grouped correctly"""
    matched_sources = [{"source": "ref_single.txt", "similarity": 85.0}]
    sentence_matches = [
        {"index": 1, "submission_sentence": "S1", "source": "ref_single.txt", "similarity": 85.0, "severity": "HIGH"},
        {"index": 2, "submission_sentence": "S2", "source": "ref_single.txt", "similarity": 75.0, "severity": "HIGH"},
        {"index": 3, "submission_sentence": "S3", "source": "ref_single.txt", "similarity": 65.0, "severity": "MODERATE"}
    ]
    grouped = group_matches_by_source(matched_sources, sentence_matches)

    assert len(grouped) == 1
    assert grouped[0]["source"] == "ref_single.txt"
    assert grouped[0]["flagged_sentences"] == 3
    assert grouped[0]["highest_match"] == 85.0


def test_7_empty_matches_handling():
    """TEST 7 — No matches handling (empty lists)"""
    detection_result = {"document_similarity": 0.0, "matched_sources": [], "sentence_matches": [], "statistics": {"total_sentences": 10}}
    scoring_result = {"final_score": 0.0, "risk_level": "LOW", "evidence_strength": "LOW", "statistics": {"total_sentences": 10}, "top_sources": []}

    result = build_evidence(detection_result, scoring_result)

    assert result["top_matches"] == []
    assert result["sources"] == []
    assert result["evidence_summary"]["flagged_sentences"] == 0
    assert result["evidence_summary"]["highest_similarity"] == 0.0


def test_8_exact_match_100_percent():
    """TEST 8 — Exact match 100% similarity severity and explanation"""
    sentence_matches = [
        {"index": 0, "submission_sentence": "Exact copy sentence.", "reference_sentence": "Exact copy sentence.", "source": "original.txt", "similarity": 100.0, "severity": "HIGH"}
    ]
    detection_result = {
        "document_similarity": 100.0,
        "matched_sources": [{"source": "original.txt", "similarity": 100.0}],
        "sentence_matches": sentence_matches,
        "statistics": {"total_sentences": 1, "high_similarity_sentences": 1, "moderate_similarity_sentences": 0, "flagged_sentence_percentage": 100.0, "sources_matched": 1}
    }
    scoring_result = {
        "final_score": 100.0,
        "risk_level": "CRITICAL",
        "evidence_strength": "HIGH",
        "statistics": {"total_sentences": 1, "high_similarity_sentences": 1, "moderate_similarity_sentences": 0, "flagged_sentence_percentage": 100.0, "sources_matched": 1},
        "top_sources": [{"source": "original.txt", "similarity": 100.0}]
    }

    result = build_evidence(detection_result, scoring_result)

    assert result["top_matches"][0]["similarity"] == 100.0
    assert result["top_matches"][0]["severity"] == "HIGH"
    assert any("exact or near-exact match" in bullet.lower() for bullet in result["why_flagged"])


def test_9_ollama_context_structure():
    """TEST 9 — Ollama context structure validation"""
    detection_result = load_fixture("sample_detection_result.json")
    scoring_result = {
        "final_score": 71.6,
        "risk_level": "HIGH",
        "evidence_strength": "HIGH",
        "statistics": {"total_sentences": 20, "sources_matched": 2, "flagged_sentence_percentage": 40.0},
        "top_sources": [{"source": "ai_fundamentals.txt", "similarity": 91.2}]
    }

    result = build_evidence(detection_result, scoring_result)
    ctx = result["analysis_context"]

    assert "score" in ctx
    assert "risk_level" in ctx
    assert "document_similarity" in ctx
    assert "flagged_sentence_percentage" in ctx
    assert "sources_matched" in ctx
    assert "top_sources" in ctx
    assert "top_evidence" in ctx
    assert len(ctx["top_evidence"]) <= 5


def test_10_determinism():
    """TEST 10 — Determinism across multiple executions"""
    detection_result = load_fixture("sample_detection_result.json")
    scoring_result = {
        "final_score": 71.6,
        "risk_level": "HIGH",
        "evidence_strength": "HIGH",
        "statistics": {"total_sentences": 20, "sources_matched": 2, "flagged_sentence_percentage": 40.0},
        "top_sources": [{"source": "ai_fundamentals.txt", "similarity": 91.2}]
    }

    results = [build_evidence(detection_result, scoring_result) for _ in range(10)]
    first_result = results[0]

    for r in results[1:]:
        assert r == first_result


def test_11_performance():
    """TEST 11 — Performance execution speed < 50ms"""
    detection_result = load_fixture("sample_detection_result.json")
    scoring_result = {
        "final_score": 71.6,
        "risk_level": "HIGH",
        "evidence_strength": "HIGH",
        "statistics": {"total_sentences": 20, "sources_matched": 2, "flagged_sentence_percentage": 40.0},
        "top_sources": [{"source": "ai_fundamentals.txt", "similarity": 91.2}]
    }

    start = time.perf_counter()
    for _ in range(100):
        build_evidence(detection_result, scoring_result)

    elapsed_ms = (time.perf_counter() - start) * 1000
    avg_ms = elapsed_ms / 100

    assert avg_ms < 50.0, f"Evidence engine execution too slow: {avg_ms:.2f} ms"
