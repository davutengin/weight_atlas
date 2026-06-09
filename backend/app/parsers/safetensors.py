import json
import struct
from pathlib import Path
from app.parsers.base import ModelAdapter
from app.models.schemas import ModelOverview, TensorInfo, ModelMetadata, ModelFormat


class SafeTensorsAdapter(ModelAdapter):
    @classmethod
    def can_handle(cls, path: Path) -> bool:
        return path.suffix.lower() == ".safetensors"

    def _read_header(self) -> dict:
        with open(self.path, "rb") as f:
            header_size_bytes = f.read(8)
            header_size = struct.unpack("<Q", header_size_bytes)[0]
            header_json = f.read(header_size)
            return json.loads(header_json)

    def get_vocab(self) -> list[str]:
        tokenizer_path = self.path.parent / "tokenizer.json"
        if not tokenizer_path.exists():
            return []
        try:
            with open(tokenizer_path, encoding="utf-8") as f:
                tok = json.load(f)
            vocab_dict: dict = (
                tok.get("model", {}).get("vocab")
                or tok.get("vocab")
                or {}
            )
            if not vocab_dict:
                return []
            tokens = [''] * len(vocab_dict)
            for token, idx in vocab_dict.items():
                if idx < len(tokens):
                    tokens[idx] = token
            return tokens
        except Exception:
            return []

    def get_tensors(self) -> list[TensorInfo]:
        header = self._read_header()
        tensors = []
        for name, info in header.items():
            if name == "__metadata__":
                continue
            shape = info.get("shape", [])
            dtype = info.get("dtype", "F32")
            param_count = self._param_count(shape)
            size_bytes = param_count * self._dtype_size(dtype)
            tensors.append(TensorInfo(
                name=name,
                shape=shape,
                dtype=dtype,
                param_count=param_count,
                size_bytes=size_bytes,
                layer_path=self._layer_path(name),
            ))
        return tensors

    def get_metadata(self) -> list[ModelMetadata]:
        header = self._read_header()
        meta = header.get("__metadata__", {})
        return [
            ModelMetadata(key=k, value=v, value_type=type(v).__name__)
            for k, v in meta.items()
        ]

    def get_overview(self) -> ModelOverview:
        tensors = self.get_tensors()
        metadata = self.get_metadata()
        meta_map = {m.key: m.value for m in metadata}

        total_params = sum(t.param_count for t in tensors)
        arch = meta_map.get("architecture") or meta_map.get("model_type") or _infer_arch(tensors)
        vocab_size = _find_vocab_size(tensors, meta_map)

        # Detect vocab size from tokenizer.json if not already found
        if vocab_size is None:
            tokenizer_path = self.path.parent / "tokenizer.json"
            if tokenizer_path.exists():
                try:
                    with open(tokenizer_path, encoding="utf-8") as f:
                        tok = json.load(f)
                    vocab_dict = (
                        tok.get("model", {}).get("vocab") or tok.get("vocab") or {}
                    )
                    if vocab_dict:
                        vocab_size = len(vocab_dict)
                except Exception:
                    pass

        return ModelOverview(
            id=str(self.path),
            name=self.path.stem,
            path=str(self.path),
            format=ModelFormat.safetensors,
            file_size=self.path.stat().st_size,
            tensor_count=len(tensors),
            param_count=total_params,
            quantization=meta_map.get("quantization"),
            architecture=arch,
            vocab_size=vocab_size,
            metadata=metadata,
        )


def _infer_arch(tensors: list[TensorInfo]) -> str | None:
    names = {t.name for t in tensors}
    if any("transformer" in n for n in names):
        return "transformer"
    if any("diffusion" in n for n in names):
        return "diffusion"
    return None


def _find_vocab_size(tensors: list[TensorInfo], meta: dict) -> int | None:
    if "vocab_size" in meta:
        try:
            return int(meta["vocab_size"])
        except (ValueError, TypeError):
            pass
    for t in tensors:
        if "embed_tokens" in t.name or "wte" in t.name:
            return t.shape[0] if t.shape else None
    return None
