from pydantic import BaseModel
from typing import Any, Optional
from enum import Enum


class ModelFormat(str, Enum):
    safetensors = "safetensors"
    gguf = "gguf"
    mlx = "mlx"
    unknown = "unknown"


class TensorInfo(BaseModel):
    name: str
    shape: list[int]
    dtype: str
    param_count: int
    size_bytes: int
    layer_path: str


class ModelMetadata(BaseModel):
    key: str
    value: Any
    value_type: str


class ModelOverview(BaseModel):
    id: str
    name: str
    path: str
    format: ModelFormat
    file_size: int
    tensor_count: int
    param_count: int
    quantization: Optional[str] = None
    architecture: Optional[str] = None
    vocab_size: Optional[int] = None
    metadata: list[ModelMetadata] = []


class TensorListResponse(BaseModel):
    tensors: list[TensorInfo]
    total: int
    total_params: int
    total_size: int
    page: int
    page_size: int


class AtlasNode(BaseModel):
    name: str
    path: str
    size: int
    param_count: int
    children: list["AtlasNode"] = []
    tensor: Optional[TensorInfo] = None


AtlasNode.model_rebuild()


class SearchResult(BaseModel):
    tensors: list[TensorInfo]
    query: str
    total: int


class LoadModelRequest(BaseModel):
    path: str


class LoadModelResponse(BaseModel):
    model_id: str
    overview: ModelOverview
