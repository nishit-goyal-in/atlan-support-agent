"""Minimal FastAPI test endpoint for Vercel deployment testing."""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="Atlan Support Agent - Test")

@app.get("/")
async def root():
    return JSONResponse({
        "message": "Atlan Support Agent - Vercel Test Deployment",
        "status": "running",
        "version": "test-minimal"
    })

@app.get("/health")
async def health():
    return {"status": "healthy"}

# Export for Vercel
handler = app