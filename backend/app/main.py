"""
ChronoVideo - Main Application Entrypoint
FastAPI server serving the REST API and the Frontend Workspace.
"""

import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from .config import APP_TITLE, APP_VERSION, APP_DESCRIPTION, FRONTEND_DIR, PORT, HOST
from .api.router import router

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=APP_DESCRIPTION
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(router)

# Mount Frontend static files
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
@app.get("/index.html")
@app.get("/workspace")
@app.get("/detail")
@app.get("/details")
@app.get("/health")
@app.get("/status")
async def serve_index():
    """Serves the ChronoVideo frontend workspace."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "status": "online",
        "service": APP_TITLE,
        "docs": "/docs",
        "frontend": "index.html pending"
    }


@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Catch-all for frontend paths so users never see 'detail: Not Found'."""
    if not request.url.path.startswith("/api/"):
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
    return JSONResponse(status_code=404, content={"detail": "Not Found"})


def run():
    import uvicorn
    if sys.platform == "win32":
        import io
        if hasattr(sys.stdout, "buffer"):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "buffer"):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    print(f"\n{'='*70}")
    print(f"   [ChronoVideo Platform is LIVE]")
    print(f"   Web Workspace: http://localhost:{PORT}")
    print(f"   API Swagger Docs: http://localhost:{PORT}/docs")
    print(f"{'='*70}\n")
    uvicorn.run("backend.app.main:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    run()
