import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from main import app
from ai.ollama_service import (
    generate_explanation,
    check_ollama_status,
    select_model,
    build_system_prompt,
    build_user_prompt,
    clean_json_response,
    PRIMARY_MODEL,
    FALLBACK_MODEL
)

client = TestClient(app)


def load_fixture(name: str) -> dict:
    fixture_path = Path(__file__).parent / "fixtures" / name
    with open(fixture_path, "r", encoding="utf-8") as f:
        return json.load(f)


MOCK_VALID_AI_RESPONSE = {
    "summary": "Significant textual overlap was detected with the local reference corpus.",
    "why_flagged": [
        "The document has a similarity score of 71.6%.",
        "55.0% of sentences contain moderate or high similarity.",
        "The strongest detected match is 94.0%."
    ],
    "key_findings": [
        {
            "source": "ai_fundamentals.txt",
            "reason": "Strong sentence-level overlap detected",
            "similarity": 91.2
        }
    ],
    "recommendations": [
        "Review the highlighted sections.",
        "Add citations where external material was used.",
        "Rewrite copied passages in your own words."
    ]
}


def make_mock_response(status_code: int, response_dict: dict):
    mock_resp = MagicMock()
    mock_resp.status = status_code
    mock_resp.read.return_value = json.dumps(response_dict).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    return mock_resp


def test_1_valid_gemma_response_parsing():
    """TEST 1 — Valid Gemma response parsing and Pydantic validation"""
    context = load_fixture("sample_evidence_result.json")
    mock_resp = make_mock_response(200, {"response": json.dumps(MOCK_VALID_AI_RESPONSE)})

    with patch("ai.ollama_service.check_ollama_status", return_value=(True, ["gemma3:4b"])):
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = generate_explanation(context)

            assert result["success"] is True
            assert result["model"] == "gemma3:4b"
            assert result["result"]["summary"] == MOCK_VALID_AI_RESPONSE["summary"]
            assert len(result["result"]["why_flagged"]) == 3
            assert result["result"]["key_findings"][0]["source"] == "ai_fundamentals.txt"


def test_2_ollama_unavailable():
    """TEST 2 — Ollama service unavailable error handling"""
    context = load_fixture("sample_evidence_result.json")

    with patch("ai.ollama_service.check_ollama_status", return_value=(False, [])):
        result = generate_explanation(context)

        assert result["success"] is False
        assert "service is unavailable" in result["error"]


def test_3_model_fallback_to_llama():
    """TEST 3 — Fallback to llama3.2:3b when gemma3:4b is missing"""
    available_models = ["llama3.2:3b", "qwen2.5-coder:3b"]
    selected = select_model(available_models)
    assert selected == "llama3.2:3b"


def test_4_invalid_json_response():
    """TEST 4 — Invalid non-JSON AI response handling"""
    context = load_fixture("sample_evidence_result.json")
    mock_resp = make_mock_response(200, {"response": "This is plain text not JSON"})

    with patch("ai.ollama_service.check_ollama_status", return_value=(True, ["gemma3:4b"])):
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = generate_explanation(context)

            assert result["success"] is False
            assert "invalid response" in result["error"]


def test_5_clean_and_low_risk_prompt_construction():
    """TEST 5 — Prompt construction for low risk / clean document"""
    low_risk_ctx = {
        "score": 5.0,
        "risk_level": "LOW",
        "document_similarity": 5.0,
        "flagged_sentence_percentage": 0.0,
        "sources_matched": 0,
        "highest_similarity": 0.0,
        "top_sources": [],
        "top_evidence": []
    }
    user_prompt = build_user_prompt(low_risk_ctx)
    assert "Risk Level: LOW" in user_prompt
    assert "None" in user_prompt


def test_6_high_risk_prompt_construction():
    """TEST 6 — Prompt construction for high risk document"""
    high_risk_ctx = load_fixture("sample_evidence_result.json")
    user_prompt = build_user_prompt(high_risk_ctx)

    assert "Plagiarism Score: 71.6%" in user_prompt
    assert "Risk Level: HIGH" in user_prompt
    assert "ai_fundamentals.txt" in user_prompt


def test_7_hallucination_and_grounding_system_prompt():
    """TEST 7 — Verify strict system prompt grounding rules"""
    system_prompt = build_system_prompt()
    assert "Do NOT invent sources" in system_prompt
    assert "Do NOT make claims of intentional cheating" in system_prompt
    assert "cautious academic language" in system_prompt


def test_8_api_endpoint_post_explain():
    """TEST 8 — POST /api/ai/explain HTTP route integration"""
    context = load_fixture("sample_evidence_result.json")
    mock_resp = make_mock_response(200, {"response": json.dumps(MOCK_VALID_AI_RESPONSE)})

    with patch("ai.ollama_service.check_ollama_status", return_value=(True, ["gemma3:4b"])):
        with patch("urllib.request.urlopen", return_value=mock_resp):
            response = client.post("/api/ai/explain", json={"analysis_context": context})

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["model"] == "gemma3:4b"
            assert data["result"]["summary"] == MOCK_VALID_AI_RESPONSE["summary"]


def test_9_markdown_json_cleaning():
    """TEST 9 — Markdown codeblock stripping helper"""
    wrapped = "```json\n{\"summary\": \"Test\"}\n```"
    cleaned = clean_json_response(wrapped)
    assert cleaned == "{\"summary\": \"Test\"}"


def test_10_live_ollama_call_if_running():
    """TEST 10 — Optional live Ollama call test if Ollama server is active"""
    is_available, models = check_ollama_status()
    if not is_available:
        pytest.skip("Local Ollama service not currently active")

    context = load_fixture("sample_evidence_result.json")
    result = generate_explanation(context)
    assert "success" in result
