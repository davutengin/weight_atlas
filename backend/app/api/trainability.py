from fastapi import APIRouter, HTTPException
from app.models.schemas import TrainabilityAnalysis
from app.analyzers.trainability import analyze
from app.api.models import _get_cached

router = APIRouter(prefix="/api/models", tags=["trainability"])


@router.get("/{model_id}/trainability", response_model=TrainabilityAnalysis)
def get_trainability(model_id: str):
    adapter = _get_cached(model_id)
    try:
        overview = adapter.get_overview()
        tensors  = adapter.get_tensors()
        return analyze(tensors, overview)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
