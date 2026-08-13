"""
Module 2 — Text Preprocessing & Normalization
Veritas AI Plagiarism Checker

Provides a clean, deterministic, reusable text preprocessing pipeline.
Maintains dual representation (Original vs Processed):
- Original text/sentences: Preserved for UI display, highlighting, evidence, and report generation.
- Processed text/sentences/tokens: Prepared for Module 3 (TF-IDF, cosine similarity, matching).

Output Contract (Module 2 -> Module 3):
{
  "filename": str,
  "source_type": str,
  "original_text": str,
  "normalized_text": str,
  "original_sentences": List[str],
  "clean_sentences": List[str],
  "sentences": List[{
      "index": int,
      "original": str,
      "clean": str,
      "tokens": List[str]
  }],
  "tokens": List[str],
  "tokens_without_stopwords": List[str],
  "sentence_count": int,
  "word_count": int,
  "character_count": int
}
"""

import re
import unicodedata
from typing import Dict, Any, List, Optional, Set

# Standard English stopwords set (lightweight, zero external model download dependency)
DEFAULT_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've",
    "were", "weren't", "what", "what's", "when", "when's", "where", "where's",
    "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your",
    "yours", "yourself", "yourselves"
}


def validate_input(doc: Any) -> Dict[str, Any]:
    """
    Validates input document payload from Module 1.
    Raises clean ValueError application exception if validation fails.
    """
    if doc is None or not isinstance(doc, dict):
        raise ValueError("Invalid document input: Expected a dictionary object.")
    
    if "text" not in doc:
        raise ValueError("Invalid document input: Missing required 'text' field.")
    
    text = doc["text"]
    if not isinstance(text, str):
        raise ValueError("Invalid document input: 'text' field must be a string.")
    
    if not text.strip():
        raise ValueError("Invalid document input: 'text' field cannot be empty.")
    
    return doc


def normalize_unicode(text: str) -> str:
    """
    Applies conservative Unicode normalization (NFC) to standardise characters.
    """
    return unicodedata.normalize("NFC", text)


def normalize_whitespace(text: str) -> str:
    """
    Normalizes spaces, tabs, and line breaks into standard whitespace representation.
    """
    return re.sub(r'\s+', ' ', text).strip()


def split_sentences(text: str) -> List[str]:
    """
    Splits document text into logical sentences while preserving original content.
    Handles multi-line texts and punctuation bounds accurately.
    """
    if not text or not text.strip():
        return []
    
    # Split paragraphs first to preserve logical chunking
    raw_paragraphs = [p.strip() for p in re.split(r'[\r\n]+', text) if p.strip()]
    sentences = []
    
    for para in raw_paragraphs:
        # Split on terminal punctuation followed by space or boundary
        chunks = re.split(r'(?<=[.!?])\s+', para)
        for chunk in chunks:
            cleaned = chunk.strip()
            if cleaned:
                sentences.append(cleaned)
                
    return sentences


def tokenize(text: str) -> List[str]:
    """
    Extracts lowercased word tokens from text.
    """
    return re.findall(r'\b\w+\b', text.lower())


def remove_stopwords(tokens: List[str], custom_stopwords: Optional[Set[str]] = None) -> List[str]:
    """
    Filters out English stopwords from a list of tokens.
    """
    stopwords = custom_stopwords if custom_stopwords is not None else DEFAULT_STOPWORDS
    return [t for t in tokens if t.lower() not in stopwords]


def clean_sentence(sentence: str) -> str:
    """
    Creates a clean, normalized string for a single sentence:
    Unicode normalized, lowercased, punctuation removed, stopwords removed, single spaced.
    Fallback to all lowercased tokens if stopword filtering leaves sentence empty.
    """
    norm = normalize_unicode(sentence)
    lowered = norm.lower()
    no_punct = re.sub(r'[^\w\s]', ' ', lowered)
    raw_tokens = re.findall(r'\b\w+\b', no_punct)
    stop_free = [t for t in raw_tokens if t not in DEFAULT_STOPWORDS]
    
    final_tokens = stop_free if stop_free else raw_tokens
    return " ".join(final_tokens)


def preprocess_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main Module 2 Preprocessing Pipeline.

    Receives Module 1 Document Dict, validates text, performs normalization, sentence splitting,
    tokenization, stopword handling, and sentence indexing.

    Returns the deterministic structured Preprocessing Result required by Module 3.
    """
    # 1. Input Validation
    validate_input(doc)

    # 2. Original Text Preservation
    original_text: str = doc["text"]
    filename: str = doc.get("filename", "document.txt")
    source_type: str = doc.get("source_type", "text")

    # 3 & 4. Unicode & Whitespace Normalization
    unicode_clean_text = normalize_unicode(original_text)
    normalized_text = normalize_whitespace(unicode_clean_text)

    # 5. Sentence Splitting (Original sentences)
    original_sentences = split_sentences(original_text)

    # 6, 7, 8, 9. Create clean sentences, clean tokens, mapping
    sentences_payload: List[Dict[str, Any]] = []
    clean_sentences_list: List[str] = []

    for idx, orig_sent in enumerate(original_sentences):
        clean_sent = clean_sentence(orig_sent)
        clean_tokens = tokenize(clean_sent)
        
        sentences_payload.append({
            "index": idx,
            "original": orig_sent,
            "clean": clean_sent,
            "tokens": clean_tokens
        })
        clean_sentences_list.append(clean_sent)

    # Global token extraction
    all_tokens = tokenize(normalized_text)
    all_tokens_no_stopwords = remove_stopwords(all_tokens)

    # Calculation of dynamic stats
    words = original_text.split()
    word_count = len(words)
    character_count = len(original_text)
    sentence_count = len(sentences_payload)

    # Return Output Contract
    return {
        "filename": filename,
        "source_type": source_type,
        "original_text": original_text,
        "normalized_text": normalized_text,
        "original_sentences": original_sentences,
        "clean_sentences": clean_sentences_list,
        "sentences": sentences_payload,
        "tokens": all_tokens,
        "tokens_without_stopwords": all_tokens_no_stopwords,
        "sentence_count": sentence_count,
        "word_count": word_count,
        "character_count": character_count
    }
