"""
MLX model format parser.
MLX models are directories containing .safetensors shards + config.json.
"""
import json
from pathlib import Path
from app.parsers.base import ModelAdapter
from app.parsers.safetensors import SafeTensorsAdapter
from app.models.schemas import ModelOverview, TensorInfo, ModelMetadata, ModelFormat


class MLXAdapter(ModelAdapter):
    @classmethod
    def can_handle(cls, path: Path) -> bool:
        if path.is_dir():
            return any(path.glob("*.safetensors")) and (path / "config.json").exists()
        return False

    def _get_shards(self) -> list[Path]:
        return sorted(self.path.glob("*.safetensors"))

    def _get_config(self) -> dict:
        config_path = self.path / "config.json"
        if config_path.exists():
            return json.loads(config_path.read_text(encoding="utf-8"))
        return {}

    def get_tensors(self) -> list[TensorInfo]:
        tensors = []
        for shard in self._get_shards():
            adapter = SafeTensorsAdapter(shard)
            tensors.extend(adapter.get_tensors())
        return tensors

    def get_metadata(self) -> list[ModelMetadata]:
        config = self._get_config()
        return [
            ModelMetadata(key=k, value=v, value_type=type(v).__name__)
            for k, v in config.items()
        ]

    def get_overview(self) -> ModelOverview:
        tensors = self.get_tensors()
        config = self._get_config()
        metadata = self.get_metadata()

        total_size = sum(s.stat().st_size for s in self._get_shards())
        total_params = sum(t.param_count for t in tensors)

        return ModelOverview(
            id=str(self.path),
            name=self.path.name,
            path=str(self.path),
            format=ModelFormat.mlx,
            file_size=total_size,
            tensor_count=len(tensors),
            param_count=total_params,
            quantization=config.get("quantization"),
            architecture=config.get("model_type"),
            vocab_size=config.get("vocab_size"),
            metadata=metadata,
        )
