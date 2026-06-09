from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.api.models import _get_cached
from app.analyzers.graph import build_mermaid
from app.models.schemas import TensorInfo

router = APIRouter(prefix="/api/models", tags=["graph"])


class GraphResponse(BaseModel):
    diagram: str
    node_map: dict[str, TensorInfo]  # mermaid node id → tensor info


@router.get("/{model_id}/graph", response_model=GraphResponse)
def get_graph(model_id: str):
    adapter = _get_cached(model_id)
    try:
        tensors  = adapter.get_tensors()
        overview = adapter.get_overview()
        diagram, node_map = build_mermaid(tensors, overview)
        return GraphResponse(diagram=diagram, node_map=node_map)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
