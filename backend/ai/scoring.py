from typing import Dict, Any, List

# Configurable Scoring Formula Weights
DOCUMENT_WEIGHT = 0.50
FLAGGED_SENTENCE_WEIGHT = 0.30
SOURCE_BREADTH_WEIGHT = 0.20

# Transparent Risk Band Thresholds
LOW_MAX = 25.0
MODERATE_MAX = 50.0
HIGH_MAX = 75.0


def calculate_source_breadth(sources_matched: int) -> float:
    """
    Normalizes matched reference source count to a 0-100 scale:
    - 0 sources  -> 0.0
    - 1 source   -> 40.0
    - 2 sources  -> 70.0
    - 3+ sources -> 100.0
    """
    if sources_matched <= 0:
        return 0.0
    elif sources_matched == 1:
        return 40.0
    elif sources_matched == 2:
        return 70.0
    else:
        return 100.0


def classify_risk(final_score: float) -> str:
    """
    Classifies final score into transparent risk bands:
    - 0.0 to 25.0    -> LOW
    - >25.0 to 50.0  -> MODERATE
    - >50.0 to 75.0  -> HIGH
    - >75.0 to 100   -> CRITICAL
    """
    if final_score <= LOW_MAX:
        return "LOW"
    elif final_score <= MODERATE_MAX:
        return "MODERATE"
    elif final_score <= HIGH_MAX:
        return "HIGH"
    else:
        return "CRITICAL"


def calculate_evidence_strength(
    document_sim: float,
    flagged_pct: float,
    sources_matched: int,
    sentence_matches: List[Dict[str, Any]]
) -> str:
    """
    Determines evidence strength transparently based on match density:
    - HIGH: Multiple matched sources AND substantial flagged content (>=30%) AND document sim >= 50%
            OR very high similarity (>=80%) with multiple high similarity sentence matches.
    - MEDIUM: Moderate evidence (document_sim >= 25% OR flagged_pct >= 15% OR sources_matched >= 1).
    - LOW: Little or no evidence.
    """
    high_sim_sentences = sum(
        1 for m in sentence_matches
        if m.get("severity") == "HIGH" or m.get("similarity", 0) >= 75.0
    )

    if (sources_matched >= 2 and flagged_pct >= 30.0 and document_sim >= 50.0) or (document_sim >= 80.0 and high_sim_sentences >= 2):
        return "HIGH"
    elif document_sim >= 25.0 or flagged_pct >= 15.0 or sources_matched >= 1 or high_sim_sentences >= 1:
        return "MEDIUM"
    else:
        return "LOW"


def validate_detection_result(detection_result: Dict[str, Any]) -> None:
    """
    Validates input detection result structure from Module 3.
    Raises ValueError if input is malformed or invalid.
    """
    if not isinstance(detection_result, dict):
        raise ValueError("Invalid detection_result: input must be a dictionary.")

    stats = detection_result.get("statistics")
    if stats is not None and isinstance(stats, dict):
        total_sentences = stats.get("total_sentences", 0)
        if total_sentences == 0:
            raise ValueError("Invalid document: no analyzable sentences.")


def calculate_score(detection_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main public API for Module 4: Scoring & Risk Classification Engine.
    
    Transforms Module 3 Detection Evidence into a transparent scoring summary:
    - Calculates 50/30/20 weighted plagiarism score
    - Classifies risk level (LOW, MODERATE, HIGH, CRITICAL)
    - Determines evidence strength (LOW, MEDIUM, HIGH)
    - Constructs transparent score breakdown
    - Returns JSON-compatible Module 4 output contract
    """
    validate_detection_result(detection_result)

    # Extract & validate document similarity
    raw_doc_sim = detection_result.get("document_similarity", 0.0)
    document_sim = max(0.0, min(100.0, float(raw_doc_sim)))

    # Extract & validate matched sources
    matched_sources = detection_result.get("matched_sources", [])
    if not isinstance(matched_sources, list):
        matched_sources = []

    # Extract statistics
    stats = detection_result.get("statistics", {})
    if not isinstance(stats, dict):
        stats = {}

    total_sentences = stats.get("total_sentences", 0)
    sources_matched = stats.get("sources_matched", len(matched_sources))

    # Extract or calculate flagged sentence percentage
    raw_flagged_pct = stats.get("flagged_sentence_percentage")
    if raw_flagged_pct is None:
        high_cnt = stats.get("high_similarity_sentences", 0)
        mod_cnt = stats.get("moderate_similarity_sentences", 0)
        flagged_cnt = high_cnt + mod_cnt
        raw_flagged_pct = (flagged_cnt / total_sentences * 100.0) if total_sentences > 0 else 0.0

    flagged_pct = max(0.0, min(100.0, float(raw_flagged_pct)))

    # Calculate normalized source breadth score
    source_breadth_score = calculate_source_breadth(sources_matched)

    # Weighted contribution calculations
    doc_contrib = document_sim * DOCUMENT_WEIGHT
    sent_contrib = flagged_pct * FLAGGED_SENTENCE_WEIGHT
    source_contrib = source_breadth_score * SOURCE_BREADTH_WEIGHT

    raw_final_score = doc_contrib + sent_contrib + source_contrib
    clamped_final_score = max(0.0, min(100.0, raw_final_score))
    final_score = round(clamped_final_score, 1)

    # Risk level classification
    risk_level = classify_risk(final_score)

    # Sentence matches & evidence strength
    sentence_matches = detection_result.get("sentence_matches", [])
    if not isinstance(sentence_matches, list):
        sentence_matches = []

    evidence_strength = calculate_evidence_strength(
        document_sim, flagged_pct, sources_matched, sentence_matches
    )

    # Construct score breakdown
    score_breakdown = {
        "document_similarity": {
            "value": round(document_sim, 1),
            "weight": DOCUMENT_WEIGHT,
            "contribution": round(doc_contrib, 1)
        },
        "flagged_sentence_percentage": {
            "value": round(flagged_pct, 1),
            "weight": FLAGGED_SENTENCE_WEIGHT,
            "contribution": round(sent_contrib, 1)
        },
        "source_breadth": {
            "value": round(source_breadth_score, 1),
            "weight": SOURCE_BREADTH_WEIGHT,
            "contribution": round(source_contrib, 1)
        }
    }

    # Construct normalized statistics output
    output_statistics = {
        "total_sentences": total_sentences,
        "high_similarity_sentences": stats.get("high_similarity_sentences", 0),
        "moderate_similarity_sentences": stats.get("moderate_similarity_sentences", 0),
        "flagged_sentence_percentage": round(flagged_pct, 1),
        "sources_matched": sources_matched
    }

    # Format top sources
    top_sources = []
    for s in matched_sources:
        if isinstance(s, dict) and "source" in s:
            top_sources.append({
                "source": s["source"],
                "similarity": round(float(s.get("similarity", 0.0)), 1)
            })

    return {
        "final_score": final_score,
        "risk_level": risk_level,
        "evidence_strength": evidence_strength,
        "score_breakdown": score_breakdown,
        "statistics": output_statistics,
        "top_sources": top_sources
    }
