from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
from pydantic import BaseModel
from app.state import model_cache
from app.parsers.data_reader import read_safetensors_slice, read_gguf_slice

router = APIRouter(prefix="/api/models", tags=["data"])

MAX_CELLS = 128  # max rows or cols per request


class TensorDataResponse(BaseModel):
    name: str
    shape: list[int]
    dtype: str
    total_rows: int
    total_cols: int
    row_offset: int
    col_offset: int
    row_count: int
    col_count: int
    data: list[list[float]]
    stats: dict[str, float]


@router.get("/{model_id}/tensor-data", response_model=TensorDataResponse)
def get_tensor_data(
    model_id: str,
    name: str = Query(...),
    row_offset: int = Query(0, ge=0),
    row_count: int = Query(64, ge=1, le=MAX_CELLS),
    col_offset: int = Query(0, ge=0),
    col_count: int = Query(64, ge=1, le=MAX_CELLS),
):
    path = Path(model_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Model path not found")

    suffix = path.suffix.lower()
    try:
        if suffix == ".safetensors":
            result = read_safetensors_slice(path, name, row_offset, row_count, col_offset, col_count)
        elif suffix == ".gguf":
            result = read_gguf_slice(path, name, row_offset, row_count, col_offset, col_count)
        else:
            raise HTTPException(status_code=400, detail=f"Tensor data reading not supported for {suffix}")
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read tensor data: {e}")

    return TensorDataResponse(
        name=result.name,
        shape=result.shape,
        dtype=result.dtype,
        total_rows=result.total_rows,
        total_cols=result.total_cols,
        row_offset=result.row_offset,
        col_offset=result.col_offset,
        row_count=result.row_count,
        col_count=result.col_count,
        data=result.data,
        stats=result.stats,
    )
