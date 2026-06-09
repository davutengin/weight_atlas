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


class LoRACandidate(BaseModel):
    module: str
    layer_count: int
    param_count: int
    priority: str  # "core" | "optional"


class FineTuningPreset(BaseModel):
    name: str
    modules: list[str]
    trainable_params: int
    adapter_size_mb: float
    pct_of_model: float


class ArchitectureFingerprint(BaseModel):
    family: str
    confidence: int
    reasons: list[str]
    hidden_size: Optional[int] = None
    layer_count: Optional[int] = None
    attention_heads: Optional[int] = None


class QLoRAAnalysis(BaseModel):
    compatible: bool
    quantization_status: str
    recommended_strategy: str
    reasons: list[str]


class ReadinessScore(BaseModel):
    score: int
    reasons: list[str]


class TrainabilityAnalysis(BaseModel):
    fingerprint: ArchitectureFingerprint
    lora_candidates: list[LoRACandidate]
    presets: list[FineTuningPreset]
    qlora: QLoRAAnalysis
    readiness: ReadinessScore
    summary: str
    recommended_preset: str
    recommended_modules: list[str]


class LoadModelRequest(BaseModel):
    path: str


class LoadModelResponse(BaseModel):
    model_id: str
    overview: ModelOverview
