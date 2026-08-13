import io
import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# Standard valid PDF binary data containing text "Hello Veritas AI Plagiarism Checker Test PDF"
VALID_PDF_BYTES = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
    b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj\n"
    b"4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
    b"5 0 obj<</Length 56>>stream\n"
    b"BT /F1 12 Tf 100 700 TD (Hello Veritas AI Plagiarism Checker Test PDF Document) Tj ET\n"
    b"endstream\n"
    b"endobj\n"
    b"xref\n"
    b"0 6\n"
    b"0000000000 65535 f \n"
    b"0000000009 00000 n \n"
    b"0000000056 00000 n \n"
    b"0000000113 00000 n \n"
    b"0000000244 00000 n \n"
    b"0000000317 00000 n \n"
    b"trailer<</Size 6/Root 1 0 R>>\n"
    b"startxref\n"
    b"423\n"
    b"%%EOF\n"
)

# PDF with no text stream (simulating image/scanned PDF)
NO_TEXT_PDF_BYTES = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
    b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Contents 4 0 R>>endobj\n"
    b"4 0 obj<</Length 0>>stream\n\nendstream\nendobj\n"
    b"xref\n"
    b"0 5\n"
    b"0000000000 65535 f \n"
    b"0000000009 00000 n \n"
    b"0000000056 00000 n \n"
    b"0000000113 00000 n \n"
    b"0000000199 00000 n \n"
    b"trailer<</Size 5/Root 1 0 R>>\n"
    b"startxref\n"
    b"249\n"
    b"%%EOF\n"
)


def test_plain_text_success():
    """Test 1: Plain text analysis success"""
    sample_text = (
        "Artificial intelligence is transforming modern education. "
        "Machine learning allows computers to learn from data and improve performance over time."
    )
    response = client.post("/api/analyze", json={"text": sample_text, "filename": "sample.txt"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    doc = data["document"]
    assert doc["source_type"] == "text"
    assert doc["filename"] == "sample.txt"
    assert doc["word_count"] > 10
    assert doc["character_count"] == len(sample_text)
    assert doc["page_count"] is None


def test_plain_text_too_short():
    """Test 3a: Text input too short (< 50 chars)"""
    response = client.post("/api/analyze", json={"text": "Too short"})
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert "at least 50 characters" in data["error"]


def test_empty_text():
    """Test 3b: Empty text input"""
    response = client.post("/api/analyze", json={"text": "   "})
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert "Please provide some text" in data["error"]


def test_valid_pdf_upload():
    """Test 2: Valid text PDF upload"""
    files = {"file": ("assignment.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")}
    response = client.post("/api/analyze", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    doc = data["document"]
    assert doc["source_type"] == "pdf"
    assert doc["filename"] == "assignment.pdf"
    assert "Veritas AI" in doc["text"]
    assert doc["page_count"] == 1


def test_invalid_file_extension():
    """Test 4: Reject non-PDF files"""
    files = {"file": ("test.docx", io.BytesIO(b"Fake docx content"), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
    response = client.post("/api/analyze", files=files)
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert "Please upload a PDF file" in data["error"]


def test_large_file_rejection():
    """Test 5: Reject files > 10 MB"""
    large_dummy_bytes = b"%PDF-1.4 " + b"0" * (10 * 1024 * 1024 + 100)
    files = {"file": ("large.pdf", io.BytesIO(large_dummy_bytes), "application/pdf")}
    response = client.post("/api/analyze", files=files)
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert "10 MB" in data["error"]


def test_scanned_image_pdf():
    """Test 6: Reject scanned / non-readable text PDF"""
    files = {"file": ("scanned.pdf", io.BytesIO(NO_TEXT_PDF_BYTES), "application/pdf")}
    response = client.post("/api/analyze", files=files)
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert "scanned/image-based content" in data["error"]
