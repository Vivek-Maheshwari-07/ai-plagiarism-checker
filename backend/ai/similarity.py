"""
Module 3 — Plagiarism / Similarity Detection Engine
Veritas AI Plagiarism Checker

Core similarity detection engine that compares a preprocessed document against a local
reference corpus using TF-IDF + Cosine Similarity and difflib.SequenceMatcher.

Calculates real, objective similarity evidence at both document and sentence levels.
Does NOT perform final risk classification or call LLM/external APIs.

Output Contract (Module 3 -> Module 4):
{
  "document_similarity": float (0-100),
  "matched_sources": List[{ "source": str, "similarity": float }],
  "sentence_matches": List[{
      "index": int,
      "submission_sentence": str,
      "reference_sentence": str,
      "source": str,
      "tfidf_similarity": float (0-1),
      "sequence_similarity": float (0-1),
      "similarity": float (0-100),
      "severity": "HIGH" | "MODERATE" | "LOW"
  }],
  "statistics": {
      "total_sentences": int,
      "high_similarity_sentences": int,
      "moderate_similarity_sentences": int,
      "flagged_sentence_percentage": float,
      "sources_matched": int,
      "max_document_similarity": float,
      "average_sentence_similarity": float
  }
}
"""

import os
import difflib
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from ai.preprocessing import preprocess_document
except ImportError:
    from .preprocessing import preprocess_document

# Configurable detection thresholds
HIGH_MATCH_THRESHOLD: float = 0.75
MODERATE_MATCH_THRESHOLD: float = 0.50


def load_reference_corpus(reference_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Loads and preprocesses reference text documents from the local reference directory.
    Returns a list of structured reference document dictionaries.
    """
    if reference_dir is None:
        # Default path: <project_root>/data/reference_documents/
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        reference_dir = os.path.join(base_dir, "data", "reference_documents")

    if not os.path.exists(reference_dir) or not os.path.isdir(reference_dir):
        return []

    reference_corpus: List[Dict[str, Any]] = []

    for filename in sorted(os.listdir(reference_dir)):
        if filename.endswith(".txt"):
            filepath = os.path.join(reference_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                if content and content.strip():
                    doc_obj = {
                        "filename": filename,
                        "source_type": "reference",
                        "text": content
                    }
                    preprocessed = preprocess_document(doc_obj)
                    preprocessed["id"] = f"ref_{len(reference_corpus) + 1:03d}"
                    reference_corpus.append(preprocessed)
            except Exception:
                continue

    return reference_corpus


def calculate_sequence_similarity(seq1: str, seq2: str) -> float:
    """
    Calculates lexical sequence similarity using difflib.SequenceMatcher.
    """
    if not seq1 or not seq2:
        return 0.0
    return difflib.SequenceMatcher(None, seq1, seq2).ratio()


def calculate_document_similarity(
    preprocessed_submission: Dict[str, Any],
    reference_corpus: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Computes document-level similarity between submission and reference corpus using TF-IDF + Cosine Similarity.
    """
    if not reference_corpus or not preprocessed_submission.get("sentences"):
        return {
            "document_similarity": 0.0,
            "matched_sources": []
        }

    submission_clean = preprocessed_submission.get("normalized_text", "")
    if not submission_clean:
        submission_clean = " ".join(preprocessed_submission.get("clean_sentences", []))

    corpus_clean_texts = []
    for ref in reference_corpus:
        ref_text = ref.get("normalized_text", "")
        if not ref_text:
            ref_text = " ".join(ref.get("clean_sentences", []))
        corpus_clean_texts.append(ref_text)

    all_docs = [submission_clean] + corpus_clean_texts

    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        tfidf_matrix = vectorizer.fit_transform(all_docs)

        sub_vec = tfidf_matrix[0:1]
        ref_vecs = tfidf_matrix[1:]

        sim_matrix = cosine_similarity(sub_vec, ref_vecs)[0]
    except Exception:
        return {
            "document_similarity": 0.0,
            "matched_sources": []
        }

    matched_sources = []
    for idx, ref in enumerate(reference_corpus):
        sim_score = float(sim_matrix[idx])
        sim_pct = round(sim_score * 100, 1)
        matched_sources.append({
            "source": ref["filename"],
            "similarity": sim_pct
        })

    # Sort sources descending by similarity
    matched_sources.sort(key=lambda x: x["similarity"], reverse=True)

    # Top matching document similarity percentage
    max_doc_sim = matched_sources[0]["similarity"] if matched_sources else 0.0

    return {
        "document_similarity": max_doc_sim,
        "matched_sources": matched_sources[:3]  # Top-K (3) matching sources
    }


def calculate_sentence_similarity(
    preprocessed_submission: Dict[str, Any],
    reference_corpus: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Computes sentence-level similarity for each sentence in submission against all reference sentences.
    Combines TF-IDF cosine similarity and SequenceMatcher lexical similarity.
    """
    sub_sentences = preprocessed_submission.get("sentences", [])
    if not sub_sentences or not reference_corpus:
        return []

    # Flatten all reference sentences
    all_ref_sentences = []
    for ref_doc in reference_corpus:
        for ref_sent in ref_doc.get("sentences", []):
            all_ref_sentences.append({
                "source": ref_doc["filename"],
                "original": ref_sent["original"],
                "clean": ref_sent["clean"]
            })

    if not all_ref_sentences:
        return []

    sub_clean_list = [s.get("clean", s.get("original", "")) for s in sub_sentences]
    ref_clean_list = [r["clean"] for r in all_ref_sentences]

    all_sentence_texts = sub_clean_list + ref_clean_list

    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        sent_matrix = vectorizer.fit_transform(all_sentence_texts)

        num_sub = len(sub_sentences)
        sub_sent_vecs = sent_matrix[:num_sub]
        ref_sent_vecs = sent_matrix[num_sub:]

        tfidf_sim_matrix = cosine_similarity(sub_sent_vecs, ref_sent_vecs)
    except Exception:
        tfidf_sim_matrix = np.zeros((len(sub_sentences), len(all_ref_sentences)))

    sentence_matches = []

    for i, sub_sent in enumerate(sub_sentences):
        best_tfidf_sim = 0.0
        best_seq_sim = 0.0
        best_combined_sim = 0.0
        best_ref_match = None

        sub_clean_str = sub_sent.get("clean", "")
        sub_orig_str = sub_sent.get("original", "")

        for j, ref_sent in enumerate(all_ref_sentences):
            t_sim = float(tfidf_sim_matrix[i, j])
            s_sim = calculate_sequence_similarity(sub_clean_str, ref_sent["clean"])
            c_sim = max(t_sim, s_sim)

            if c_sim > best_combined_sim or best_ref_match is None:
                best_combined_sim = c_sim
                best_tfidf_sim = t_sim
                best_seq_sim = s_sim
                best_ref_match = ref_sent

        sim_pct = round(best_combined_sim * 100, 1)

        if best_combined_sim >= HIGH_MATCH_THRESHOLD:
            severity = "HIGH"
        elif best_combined_sim >= MODERATE_MATCH_THRESHOLD:
            severity = "MODERATE"
        else:
            severity = "LOW"

        sentence_matches.append({
            "index": sub_sent.get("index", i),
            "submission_sentence": sub_orig_str,
            "reference_sentence": best_ref_match["original"] if best_ref_match else "",
            "source": best_ref_match["source"] if best_ref_match else "None",
            "tfidf_similarity": round(best_tfidf_sim, 2),
            "sequence_similarity": round(best_seq_sim, 2),
            "similarity": sim_pct,
            "severity": severity
        })

    return sentence_matches


def analyze_similarity(
    preprocessed_document: Dict[str, Any],
    reference_documents: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Main Module 3 Similarity Detection Entry Point.

    Computes document-level and sentence-level similarity against reference corpus.
    Returns structured Detection Result contract for Module 4.
    """
    if reference_documents is None:
        reference_documents = load_reference_corpus()

    # Document-level analysis
    doc_analysis = calculate_document_similarity(preprocessed_document, reference_documents)
    doc_similarity = doc_analysis["document_similarity"]
    matched_sources = doc_analysis["matched_sources"]

    # Sentence-level analysis
    sentence_matches = calculate_sentence_similarity(preprocessed_document, reference_documents)

    # Statistics calculation
    total_sentences = len(sentence_matches)
    high_sim_count = sum(1 for m in sentence_matches if m["severity"] == "HIGH")
    mod_sim_count = sum(1 for m in sentence_matches if m["severity"] == "MODERATE")
    flagged_count = high_sim_count + mod_sim_count

    flagged_pct = round((flagged_count / total_sentences * 100), 1) if total_sentences > 0 else 0.0

    flagged_sources = set(
        m["source"] for m in sentence_matches
        if m["severity"] in ("HIGH", "MODERATE") and m["source"] != "None"
    )
    sources_matched_count = len(flagged_sources) if flagged_sources else len(matched_sources)

    avg_sent_sim = (
        round(sum(m["similarity"] for m in sentence_matches) / total_sentences, 1)
        if total_sentences > 0 else 0.0
    )

    statistics = {
        "total_sentences": total_sentences,
        "high_similarity_sentences": high_sim_count,
        "moderate_similarity_sentences": mod_sim_count,
        "flagged_sentence_percentage": flagged_pct,
        "sources_matched": sources_matched_count,
        "max_document_similarity": doc_similarity,
        "average_sentence_similarity": avg_sent_sim
    }

    return {
        "document_similarity": doc_similarity,
        "matched_sources": matched_sources,
        "sentence_matches": sentence_matches,
        "statistics": statistics
    }
