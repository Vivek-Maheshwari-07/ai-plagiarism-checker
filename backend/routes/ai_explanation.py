from typing import Dict, Any, Optional
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ai.ollama_service import generate_explanation

router = APIRouter()


class ExplanationApiRequest(BaseModel):
    analysis_context: Dict[str, Any] = Field(..., description="Structured analysis context from Module 5")


@router.post("/explain")
async def explain_analysis(request: ExplanationApiRequest):
    """
    Endpoint for Module 6 Local AI Explanation & Recommendation Engine.
    Consumes Module 5 analysis_context and calls local Ollama (Gemma 3 4B IT / Llama 3.2 3B).
    """
    if not request.analysis_context:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"success": False, "error": "Missing analysis_context in request payload."}
        )

    response = generate_explanation(request.analysis_context)

    if not response.get("success", False):
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=response
        )

    return response
