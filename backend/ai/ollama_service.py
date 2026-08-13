import os
import json
import re
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
PRIMARY_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
FALLBACK_MODEL = os.getenv("OLLAMA_FALLBACK_MODEL", "llama3.2:3b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "15.0"))


class KeyFinding(BaseModel):
    source: str
    reason: str
    similarity: float


class ExplanationResult(BaseModel):
    summary: str
    why_flagged: List[str]
    key_findings: List[KeyFinding]
    recommendations: List[str]


class AnalysisContext(BaseModel):
    score: float
    risk_level: str
    document_similarity: Optional[float] = 0.0
    flagged_sentence_percentage: Optional[float] = 0.0
    sources_matched: Optional[int] = 0
    highest_similarity: Optional[float] = 0.0
    top_sources: Optional[List[Dict[str, Any]]] = []
    top_evidence: Optional[List[Dict[str, Any]]] = []


def check_ollama_status(base_url: str = OLLAMA_BASE_URL) -> Tuple[bool, List[str]]:
    """
    Checks if local Ollama server is running and lists installed models.
    Returns (is_available, model_names).
    """
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", headers={"User-Agent": "Veritas-AI/1.0"})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                return True, models
    except Exception:
        pass
    return False, []


def select_model(available_models: List[str]) -> str:
    """
    Selects the best available model (gemma3:4b primary, llama3.2:3b fallback).
    """
    if not available_models:
        return PRIMARY_MODEL

    # Check for primary model
    for m in available_models:
        if PRIMARY_MODEL in m or m in PRIMARY_MODEL:
            return PRIMARY_MODEL

    # Check for fallback model
    for m in available_models:
        if FALLBACK_MODEL in m or m in FALLBACK_MODEL:
            return FALLBACK_MODEL

    # If neither exact match is found, pick first available or default
    return available_models[0] if available_models else PRIMARY_MODEL


def build_system_prompt() -> str:
    """
    Constructs system prompt forcing strict evidence grounding and JSON schema compliance.
    """
    return (
        "You are an academic integrity analysis assistant for Veritas AI.\n"
        "You are given computed plagiarism/similarity evidence from a deterministic detection system.\n"
        "Your task is to explain the provided evidence cleanly, objectively, and accurately.\n\n"
        "STRICT GROUNDING RULES:\n"
        "1. Do NOT invent sources, URLs, authors, research papers, percentages, similarity values, or sentences.\n"
        "2. Do NOT make claims of intentional cheating or plagiarism. Distinguish textual similarity from plagiarism.\n"
        "3. Use cautious academic language ('similarity detected', 'review recommended', 'textual overlap').\n"
        "4. Rely ONLY on the supplied evidence context.\n"
        "5. Output ONLY valid JSON matching this exact structure without markdown formatting or codeblocks:\n"
        "{\n"
        '  "summary": "High-level summary of findings...",\n'
        '  "why_flagged": ["Reason 1", "Reason 2"],\n'
        '  "key_findings": [{"source": "filename.txt", "reason": "Reason description", "similarity": 90.0}],\n'
        '  "recommendations": ["Action item 1", "Action item 2"]\n'
        "}"
    )


def build_user_prompt(ctx: Dict[str, Any]) -> str:
    """
    Formats the evidence context into a structured user prompt.
    """
    score = ctx.get("score", 0.0)
    risk_level = ctx.get("risk_level", "LOW")
    doc_sim = ctx.get("document_similarity", 0.0)
    flagged_pct = ctx.get("flagged_sentence_percentage", 0.0)
    sources_cnt = ctx.get("sources_matched", 0)
    highest_sim = ctx.get("highest_similarity", 0.0)

    top_sources = ctx.get("top_sources", [])
    top_evidence = ctx.get("top_evidence", [])

    sources_str = ""
    if top_sources:
        for s in top_sources[:3]:
            sources_str += f"- {s.get('source', 'Unknown')}: {s.get('similarity', 0.0)}% similarity\n"
    else:
        sources_str = "None\n"

    evidence_str = ""
    if top_evidence:
        for e in top_evidence[:5]:
            evidence_str += (
                f"- [{e.get('source', 'Unknown')} | {e.get('similarity', 0.0)}% sim]\n"
                f'  Submitted: "{e.get("submitted_text", "")}"\n'
                f'  Matched:   "{e.get("matched_text", "")}"\n'
            )
    else:
        evidence_str = "No specific flagged sentences.\n"

    return (
        f"EVIDENCE SUMMARY:\n"
        f"- Plagiarism Score: {score:.1f}%\n"
        f"- Risk Level: {risk_level}\n"
        f"- Document Similarity: {doc_sim:.1f}%\n"
        f"- Flagged Sentence Percentage: {flagged_pct:.1f}%\n"
        f"- Matched Sources Count: {sources_cnt}\n"
        f"- Highest Sentence Match: {highest_sim:.1f}%\n\n"
        f"TOP SOURCES MATCHED:\n"
        f"{sources_str}\n"
        f"TOP EVIDENCE MATCHES:\n"
        f"{evidence_str}\n"
        f"Please explain why this document was flagged, summarize key findings per source, "
        f"and provide practical academic integrity recommendations."
    )


def clean_json_response(response_text: str) -> str:
    """
    Strips markdown codeblock tags (```json ... ```) if the model returned wrapped JSON.
    """
    text = response_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def generate_explanation(
    analysis_context: Dict[str, Any],
    base_url: str = OLLAMA_BASE_URL,
    override_model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Primary API for Module 6: Local AI Explanation Engine.
    Communicates with local Ollama server (Gemma 3 4B IT / Llama 3.2 3B).
    Returns structured explanation payload.
    """
    if not isinstance(analysis_context, dict):
        return {"success": False, "error": "Invalid request payload. Expected analysis_context object."}

    # Step 1: Check Ollama availability & discover installed models
    is_available, installed_models = check_ollama_status(base_url)
    if not is_available:
        return {
            "success": False,
            "error": "Local AI service is unavailable. Please ensure Ollama is running."
        }

    # Step 2: Select appropriate model
    model = override_model or select_model(installed_models)

    # Step 3: Build prompts
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(analysis_context)

    # Step 4: Send request to Ollama
    payload = {
        "model": model,
        "prompt": user_prompt,
        "system": system_prompt,
        "stream": False,
        "format": "json"
    }

    try:
        req = urllib.request.Request(
            f"{base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "Veritas-AI/1.0"}
        )

        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            if resp.status != 200:
                return {"success": False, "error": f"Ollama service returned HTTP status {resp.status}."}

            resp_data = json.loads(resp.read().decode("utf-8"))
            raw_response = resp_data.get("response", "")

            if not raw_response:
                return {"success": False, "error": "The AI returned an empty response."}

            # Step 5: Clean and parse JSON response
            cleaned_text = clean_json_response(raw_response)
            parsed_json = json.loads(cleaned_text)

            # Step 6: Validate against Pydantic schema
            validated_result = ExplanationResult(**parsed_json)

            return {
                "success": True,
                "model": model,
                "result": validated_result.model_dump()
            }

    except urllib.error.URLError:
        return {"success": False, "error": "Configured local AI model is unavailable."}
    except json.JSONDecodeError:
        return {"success": False, "error": "The AI returned an invalid response."}
    except Exception as e:
        return {"success": False, "error": f"AI service error: {str(e)}"}
