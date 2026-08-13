from typing import Optional
from fastapi import APIRouter, File, UploadFile, Request, status
from fastapi.responses import JSONResponse

from services.document import validate_text, extract_text_from_pdf, create_document

router = APIRouter()

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB in bytes


@router.post("/analyze")
async def analyze_document(
    request: Request,
    file: Optional[UploadFile] = File(None)
):
    """
    Endpoint for Document Input & Extraction (Module 1).
    Supports:
    1. JSON request with plain text body (content-type: application/json)
    2. File upload with PDF file (content-type: multipart/form-data)
    """
    content_type = request.headers.get("content-type", "").lower()

    # Case 1: PDF File Upload via multipart/form-data
    if "multipart/form-data" in content_type or file is not None:
        if not file:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"success": False, "error": "No file provided. Please upload a PDF file."}
            )

        filename = file.filename or "uploaded_document.pdf"
        
        # Extension validation
        if not filename.lower().endswith(".pdf"):
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"success": False, "error": "Please upload a PDF file."}
            )

        file_bytes = await file.read()

        # File size validation (Max 10 MB)
        if len(file_bytes) > MAX_FILE_SIZE:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"success": False, "error": "File size must be less than 10 MB."}
            )

        try:
            text, page_count = extract_text_from_pdf(file_bytes)
            doc = create_document(
                text=text,
                filename=filename,
                source_type="pdf",
                page_count=page_count
            )
            return {"success": True, "document": doc}
        except ValueError as ve:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"success": False, "error": str(ve)}
            )
        except Exception:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"success": False, "error": "Unable to read this PDF. Please upload a valid text-based PDF."}
            )

    # Case 2: Plain Text Input via JSON
    try:
        data = await request.json()
        raw_text = data.get("text", "") if isinstance(data, dict) else ""
        filename = data.get("filename") if isinstance(data, dict) and data.get("filename") else "pasted_document.txt"
        
        validated_text = validate_text(raw_text)
        doc = create_document(
            text=validated_text,
            filename=filename,
            source_type="text",
            page_count=None
        )
        return {"success": True, "document": doc}
    except ValueError as ve:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"success": False, "error": str(ve)}
        )
    except Exception:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"success": False, "error": "Invalid request payload. Please provide text to analyze."}
        )
