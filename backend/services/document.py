import io
from typing import Optional, Dict, Any, Tuple
import pdfplumber


MIN_TEXT_LENGTH = 50


def validate_text(text: Optional[str]) -> str:
    """
    Validates plain text input.
    Raises ValueError with user-friendly error message if validation fails.
    """
    if not text or not text.strip():
        raise ValueError("Please provide some text to analyze.")
    
    cleaned_text = text.strip()
    if len(cleaned_text) < MIN_TEXT_LENGTH:
        raise ValueError(f"Please enter at least {MIN_TEXT_LENGTH} characters.")
    
    return cleaned_text


def extract_text_from_pdf(file_bytes: bytes) -> Tuple[str, int]:
    """
    Extracts text from a PDF file byte buffer using pdfplumber.
    Handles multi-page documents and skips empty pages safely.
    Raises ValueError if PDF is invalid, empty, or scanned image-only.
    """
    if not file_bytes:
        raise ValueError("The uploaded PDF file is empty.")
        
    try:
        pdf_file = io.BytesIO(file_bytes)
        extracted_pages = []
        page_count = 0

        with pdfplumber.open(pdf_file) as pdf:
            page_count = len(pdf.pages)
            if page_count == 0:
                raise ValueError("The uploaded PDF contains no pages.")

            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    extracted_pages.append(page_text.strip())

    except ValueError:
        raise
    except Exception:
        raise ValueError("Unable to read this PDF. Please upload a valid text-based PDF.")

    combined_text = "\n\n".join(extracted_pages).strip()

    # If no text could be extracted or text is trivial (likely scanned image)
    if not combined_text or len(combined_text) < 10:
        raise ValueError(
            "This PDF appears to contain scanned/image-based content. "
            "OCR support will be added in a future module. "
            "For now, please paste the document text."
        )

    return combined_text, page_count


def create_document(
    text: str,
    filename: str,
    source_type: str,
    page_count: Optional[int] = None
) -> Dict[str, Any]:
    """
    Constructs the standard document structure contract for Module 1.
    """
    words = text.split()
    word_count = len(words)
    character_count = len(text)

    return {
        "filename": filename,
        "source_type": source_type,
        "text": text,
        "word_count": word_count,
        "character_count": character_count,
        "page_count": page_count
    }
