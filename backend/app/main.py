from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from app.core.config import settings
from app.api import models, tensors, search, data, trainability, vocab, graph

app = FastAPI(title=settings.app_name, version=settings.version)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(models.router)
app.include_router(tensors.router)
app.include_router(search.router)
app.include_router(data.router)
app.include_router(trainability.router)
app.include_router(vocab.router)
app.include_router(graph.router)


@app.get("/api/health")
def health():
    return {"status": "ok", "version": settings.version}


# Serve frontend build in production
frontend_dist = settings.frontend_dist
if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str):
        index = frontend_dist / "index.html"
        return FileResponse(index)
