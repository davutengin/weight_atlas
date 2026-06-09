from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
from app.models.schemas import SearchResult
from app.state import model_cache

router = APIRouter(prefix="/api/models", tags=["search"])


@router.get("/{model_id}/search", response_model=SearchResult)
def search_tensors(model_id: str, q: str = Query(..., min_length=1)):
    adapter = _get_adapter(model_id)
    tensors = adapter.get_tensors()
    query = q.lower()
    matched = [t for t in tensors if query in t.name.lower()]
    return SearchResult(tensors=matched, query=q, total=len(matched))


def _get_adapter(model_id: str):
    from app.parsers.registry import get_adapter
    if model_id not in model_cache:
        path = Path(model_id)
        if path.exists():
            try:
                adapter = get_adapter(path)
                model_cache[model_id] = adapter
                return adapter
            except Exception:
                pass
        raise HTTPException(status_code=404, detail="Model not loaded")
    return model_cache[model_id]
