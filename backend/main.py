from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.analyze import router as analyze_router
from routes.ai_explanation import router as ai_explanation_router

app = FastAPI(
    title="Veritas AI Backend",
    description="AI-Based Plagiarism Checker API",
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

# Include routers
app.include_router(analyze_router, prefix="/api", tags=["Document Analysis"])
app.include_router(ai_explanation_router, prefix="/api/ai", tags=["AI Explanation"])


@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Veritas AI Backend API",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
