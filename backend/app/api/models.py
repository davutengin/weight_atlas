from fastapi import APIRouter, HTTPException
from pathlib import Path
from app.models.schemas import LoadModelRequest, LoadModelResponse, ModelOverview
from app.parsers.registry import get_adapter
from app.state import model_cache

router = APIRouter(prefix="/api/models", tags=["models"])


@router.post("/load", response_model=LoadModelResponse)
def load_model(req: LoadModelRequest):
    path = Path(req.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {req.path}")
    try:
        adapter = get_adapter(path)
        overview = adapter.get_overview()
        model_cache[overview.id] = adapter
        return LoadModelResponse(model_id=overview.id, overview=overview)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse model: {e}")


@router.get("/{model_id}/overview", response_model=ModelOverview)
def get_overview(model_id: str):
    adapter = _get_cached(model_id)
    return adapter.get_overview()


def _get_cached(model_id: str):
    from app.state import model_cache
    if model_id not in model_cache:
        # try to reload from path
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
