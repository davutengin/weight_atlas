from pathlib import Path
from app.parsers.base import ModelAdapter
from app.parsers.safetensors import SafeTensorsAdapter
from app.parsers.gguf import GGUFAdapter
from app.parsers.mlx import MLXAdapter

ADAPTERS: list[type[ModelAdapter]] = [MLXAdapter, SafeTensorsAdapter, GGUFAdapter]


def get_adapter(path: Path) -> ModelAdapter:
    for adapter_cls in ADAPTERS:
        if adapter_cls.can_handle(path):
            return adapter_cls(path)
    raise ValueError(f"Unsupported model format: {path}")
