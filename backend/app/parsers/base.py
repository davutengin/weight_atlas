from abc import ABC, abstractmethod
from pathlib import Path
from app.models.schemas import ModelOverview, TensorInfo, ModelMetadata


class ModelAdapter(ABC):
    def __init__(self, path: Path):
        self.path = path

    @abstractmethod
    def get_overview(self) -> ModelOverview:
        pass

    @abstractmethod
    def get_tensors(self) -> list[TensorInfo]:
        pass

    @abstractmethod
    def get_metadata(self) -> list[ModelMetadata]:
        pass

    @classmethod
    @abstractmethod
    def can_handle(cls, path: Path) -> bool:
        pass

    def _dtype_size(self, dtype: str) -> int:
        sizes = {
            "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1,
            "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
            "float32": 4, "float16": 2, "bfloat16": 2, "float64": 8,
            "int64": 8, "int32": 4, "int16": 2, "int8": 1, "uint8": 1,
        }
        return sizes.get(dtype, 4)

    def _param_count(self, shape: list[int]) -> int:
        result = 1
        for d in shape:
            result *= d
        return result

    def _layer_path(self, name: str) -> str:
        parts = name.replace(".", "/").split("/")
        return "/".join(parts[:-1]) if len(parts) > 1 else ""
