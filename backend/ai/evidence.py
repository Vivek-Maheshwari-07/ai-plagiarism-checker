from typing import Dict, Any, List, Tuple, Optional


TOP_MATCHES_LIMIT = 5


def validate_inputs(detection_result: Dict[str, Any], scoring_result: Dict[str, Any]) -> None:
    """
    Validates input structures from Module 3 and Module 4.
    Raises ValueError if input data is malformed.
    """
    if not isinstance(detection_result, dict):
        raise ValueError("Invalid detection_result: must be a dictionary.")
    if not isinstance(scoring_result, dict):
        raise ValueError("Invalid scoring_result: must be a dictionary.")


def format_sentence_match(match: Dict[str, Any]) -> Dict[str, Any]:
    """
    Formats a single sentence match object cleanly for frontend & evidence contracts.
    """
    similarity = float(match.get("similarity", 0.0))
    severity = match.get("severity")
    if not severity:
        severity = "HIGH" if similarity >= 75.0 else ("MODERATE" if similarity >= 50.0 else "LOW")

    return {
        "index": match.get("index", 0),
        "submitted_text": match.get("submitted_text") or match.get("submission_sentence", ""),
        "matched_text": match.get("matched_text") or match.get("reference_sentence", ""),
        "source": match.get("source", "Unknown Source"),
        "similarity": round(similarity, 1),
        "tfidf_similarity": round(float(match.get("tfidf_similarity", 0.0)), 2),
        "sequence_similarity": round(float(match.get("sequence_similarity", 0.0)), 2),
        "severity": severity
    }


def select_top_matches(sentence_matches: List[Dict[str, Any]], top_limit: int = TOP_MATCHES_LIMIT) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
    """
    Selects flagged sentences (similarity >= 50% or severity HIGH/MODERATE),
    sorts by similarity descending, and extracts top limit matches.
    Returns (top_matches, all_flagged_matches, highest_similarity).
    """
    if not isinstance(sentence_matches, list):
        return [], [], 0.0

    flagged_matches = []
    highest_similarity = 0.0

    for m in sentence_matches:
        if not isinstance(m, dict):
            continue

        sim = float(m.get("similarity", 0.0))
        severity = m.get("severity")
        
        if sim > highest_similarity:
            highest_similarity = sim

        # Include if similarity >= 50% or explicitly marked HIGH/MODERATE
        if sim >= 50.0 or severity in ("HIGH", "MODERATE"):
            formatted = format_sentence_match(m)
            flagged_matches.append(formatted)

    # Sort flagged matches by highest similarity first
    sorted_flagged = sorted(flagged_matches, key=lambda x: x["similarity"], reverse=True)
    top_matches = sorted_flagged[:top_limit]

    return top_matches, sorted_flagged, round(highest_similarity, 1)


def group_matches_by_source(matched_sources: List[Dict[str, Any]], sentence_matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Groups sentence matches by reference source file and compiles per-source statistics.
    """
    source_stats: Dict[str, Dict[str, Any]] = {}

    # Initialize from matched_sources
    if isinstance(matched_sources, list):
        for s in matched_sources:
            if isinstance(s, dict) and "source" in s:
                src_name = s["source"]
                source_stats[src_name] = {
                    "source": src_name,
                    "similarity": round(float(s.get("similarity", 0.0)), 1),
                    "flagged_sentences": 0,
                    "highest_match": 0.0
                }

    # Populate sentence match occurrences
    if isinstance(sentence_matches, list):
        for m in sentence_matches:
            if isinstance(m, dict):
                src_name = m.get("source", "Unknown Source")
                sim = round(float(m.get("similarity", 0.0)), 1)
                severity = m.get("severity")

                if src_name not in source_stats:
                    source_stats[src_name] = {
                        "source": src_name,
                        "similarity": sim,
                        "flagged_sentences": 0,
                        "highest_match": 0.0
                    }

                if sim >= 50.0 or severity in ("HIGH", "MODERATE"):
                    source_stats[src_name]["flagged_sentences"] += 1
                    if sim > source_stats[src_name]["highest_match"]:
                        source_stats[src_name]["highest_match"] = sim

    # Convert dictionary values to sorted list
    grouped_sources = list(source_stats.values())
    grouped_sources.sort(key=lambda x: (x["similarity"], x["highest_match"]), reverse=True)

    return grouped_sources


def build_why_flagged(
    score: float,
    risk_level: str,
    total_sentences: int,
    flagged_sentences: int,
    flagged_pct: float,
    sources_matched: int,
    highest_sim: float,
    top_source_name: Optional[str]
) -> List[str]:
    """
    Generates deterministic, cautious academic explanation bullets explaining WHY the document was flagged.
    Does NOT use an LLM or make unsupported accusations of plagiarism.
    """
    bullets = []

    if flagged_sentences == 0 or score <= 10.0:
        bullets.append("No significant textual overlap was detected in the reference corpus.")
        return bullets

    # Bullet 1: Score & Risk summary
    bullets.append(f"The document received a {risk_level} risk score of {score:.1f}%.")

    # Bullet 2: Affected sentence count
    if total_sentences > 0:
        bullets.append(
            f"{flagged_sentences} of {total_sentences} sentences ({flagged_pct:.1f}%) "
            f"contain moderate or high similarity."
        )

    # Bullet 3: Highest similarity match
    if highest_sim >= 99.0:
        if top_source_name:
            bullets.append(f"An exact or near-exact match (100.0%) was detected with {top_source_name}.")
        else:
            bullets.append("An exact or near-exact match (100.0%) was detected with reference material.")
    elif highest_sim > 0:
        if top_source_name:
            bullets.append(f"The strongest detected sentence match is {highest_sim:.1f}% with {top_source_name}.")
        else:
            bullets.append(f"The strongest detected sentence match is {highest_sim:.1f}% similarity.")

    # Bullet 4: Source count
    if sources_matched == 1:
        bullets.append("1 reference source contains matching content requiring review.")
    elif sources_matched > 1:
        bullets.append(f"{sources_matched} reference sources contain matching content requiring review.")

    return bullets


def build_ollama_context(
    score: float,
    risk_level: str,
    doc_sim: float,
    flagged_pct: float,
    sources_matched: int,
    highest_sim: float,
    top_sources: List[Dict[str, Any]],
    top_matches: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Constructs a compact structured context object (`analysis_context`) for Module 6 (Ollama + Gemma).
    Avoids sending heavy full-document text payloads to the LLM.
    """
    compact_evidence = []
    for m in top_matches[:5]:
        compact_evidence.append({
            "submitted_text": m.get("submitted_text", ""),
            "matched_text": m.get("matched_text", ""),
            "source": m.get("source", ""),
            "similarity": m.get("similarity", 0.0)
        })

    compact_sources = []
    for s in top_sources[:3]:
        compact_sources.append({
            "source": s.get("source", ""),
            "similarity": s.get("similarity", 0.0)
        })

    return {
        "analysis_context": {
            "score": score,
            "risk_level": risk_level,
            "document_similarity": doc_sim,
            "flagged_sentence_percentage": flagged_pct,
            "sources_matched": sources_matched,
            "highest_similarity": highest_sim,
            "top_sources": compact_sources,
            "top_evidence": compact_evidence
        }
    }


def build_evidence(detection_result: Dict[str, Any], scoring_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main public API for Module 5: Evidence & Explainability Engine.
    
    Consumes Module 3 (Detection) and Module 4 (Scoring) results and constructs:
    - Evidence Summary statistics
    - Top flagged sentence matches (capped at top 5 for UI display)
    - Reference source groupings with per-source metrics
    - Deterministic, cautious explanation bullets (`why_flagged`)
    - Structured `analysis_context` payload for Module 6 Ollama/Gemma
    """
    validate_inputs(detection_result, scoring_result)

    # Extract scoring outputs
    score = float(scoring_result.get("final_score", 0.0))
    risk_level = str(scoring_result.get("risk_level", "LOW"))
    evidence_strength = str(scoring_result.get("evidence_strength", "LOW"))
    top_sources = scoring_result.get("top_sources", [])
    if not isinstance(top_sources, list):
        top_sources = []

    # Extract detection outputs
    matched_sources = detection_result.get("matched_sources", [])
    sentence_matches = detection_result.get("sentence_matches", [])

    # Process matches & top matches selection
    top_matches, all_flagged_matches, highest_sim = select_top_matches(sentence_matches)

    # Group matches by source
    grouped_sources = group_matches_by_source(matched_sources, sentence_matches)

    # Extract statistics
    stats = scoring_result.get("statistics", {})
    if not isinstance(stats, dict):
        stats = detection_result.get("statistics", {})
        if not isinstance(stats, dict):
            stats = {}

    total_sentences = stats.get("total_sentences", 0)
    flagged_sentences = len(all_flagged_matches)
    high_sim_sentences = stats.get("high_similarity_sentences", sum(1 for m in all_flagged_matches if m["severity"] == "HIGH"))
    mod_sim_sentences = stats.get("moderate_similarity_sentences", sum(1 for m in all_flagged_matches if m["severity"] == "MODERATE"))
    
    flagged_pct = stats.get("flagged_sentence_percentage")
    if flagged_pct is None:
        flagged_pct = (flagged_sentences / total_sentences * 100.0) if total_sentences > 0 else 0.0
    flagged_pct = round(float(flagged_pct), 1)

    sources_matched = stats.get("sources_matched", len(grouped_sources))

    # Evidence summary statistics
    evidence_summary = {
        "total_sentences": total_sentences,
        "flagged_sentences": flagged_sentences,
        "high_similarity_sentences": high_sim_sentences,
        "moderate_similarity_sentences": mod_sim_sentences,
        "flagged_percentage": flagged_pct,
        "sources_matched": sources_matched,
        "highest_similarity": highest_sim
    }

    # Deterministic why_flagged bullet points
    top_source_name = top_sources[0]["source"] if top_sources and "source" in top_sources[0] else (
        grouped_sources[0]["source"] if grouped_sources else None
    )

    why_flagged = build_why_flagged(
        score=score,
        risk_level=risk_level,
        total_sentences=total_sentences,
        flagged_sentences=flagged_sentences,
        flagged_pct=flagged_pct,
        sources_matched=sources_matched,
        highest_sim=highest_sim,
        top_source_name=top_source_name
    )

    # Module 6 Ollama Context
    raw_doc_sim = detection_result.get("document_similarity", 0.0)
    doc_sim = round(float(raw_doc_sim), 1)

    ollama_context_wrapper = build_ollama_context(
        score=score,
        risk_level=risk_level,
        doc_sim=doc_sim,
        flagged_pct=flagged_pct,
        sources_matched=sources_matched,
        highest_sim=highest_sim,
        top_sources=top_sources if top_sources else grouped_sources,
        top_matches=top_matches
    )

    return {
        "score": score,
        "risk_level": risk_level,
        "evidence_strength": evidence_strength,
        "evidence_summary": evidence_summary,
        "top_matches": top_matches,
        "sources": grouped_sources,
        "why_flagged": why_flagged,
        "analysis_context": ollama_context_wrapper["analysis_context"]
    }
