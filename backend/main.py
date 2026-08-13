from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.analyze import router as analyze_router

app = FastAPI(
    title="Veritas AI Backend",
    description="AI-Based Plagiarism Checker - Module 1 Document Input & Extraction",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for local hackathon dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(analyze_router, prefix="/api", tags=["Document Analysis"])


@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Veritas AI Document Extraction API",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
