"""Vercel deployment entry point for Atlan Support Agent v2."""

import sys
from pathlib import Path

# Add the src directory to Python path
current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / 'src'))

try:
    from src.app.main import app
    
    # Export the app for Vercel
    handler = app
    
except Exception as e:
    # Fallback FastAPI app if main app fails to import
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    
    fallback_app = FastAPI(title="Atlan Support Agent - Fallback")
    
    @fallback_app.get("/")
    async def root():
        return JSONResponse({
            "error": "Application startup failed",
            "message": str(e),
            "status": "error"
        }, status_code=500)
    
    @fallback_app.get("/health")
    async def health():
        return JSONResponse({
            "status": "error",
            "error": str(e)
        }, status_code=500)
    
    handler = fallback_app