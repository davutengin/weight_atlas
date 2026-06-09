"""
GGUF parser — pure Python, no external deps.
Spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
"""
import struct
from pathlib import Path
from app.parsers.base import ModelAdapter
from app.models.schemas import ModelOverview, TensorInfo, ModelMetadata, ModelFormat

GGUF_MAGIC = b"GGUF"

GGUF_TYPE = {
    0: ("uint8", 1), 1: ("int8", 1), 2: ("uint16", 2), 3: ("int16", 2),
    4: ("uint32", 4), 5: ("int32", 4), 6: ("float32", 4), 7: ("bool", 1),
    8: ("string", 0), 9: ("array", 0), 10: ("uint64", 8), 11: ("int64", 8),
    12: ("float64", 8),
}

GGML_TYPE_INFO = {
    0: ("F32", 4), 1: ("F16", 2), 2: ("Q4_0", 0.5), 3: ("Q4_1", 0.5625),
    6: ("Q5_0", 0.625), 7: ("Q5_1", 0.6875), 8: ("Q8_0", 1.0), 9: ("Q8_1", 1.125),
    10: ("Q2_K", 0.328125), 11: ("Q3_K", 0.421875), 12: ("Q4_K", 0.5625),
    13: ("Q5_K", 0.6875), 14: ("Q6_K", 0.8125), 15: ("Q8_K", 1.09375),
    16: ("IQ2_XXS", 0.257), 17: ("IQ2_XS", 0.289), 18: ("IQ3_XXS", 0.383),
    19: ("IQ1_S", 0.189), 20: ("IQ4_NL", 0.5), 21: ("IQ3_S", 0.422),
    22: ("IQ2_S", 0.312), 23: ("IQ4_XS", 0.453), 24: ("I8", 1),
    25: ("I16", 2), 26: ("I32", 4), 27: ("I64", 8), 28: ("F64", 8),
    29: ("IQ1_M", 0.179), 30: ("BF16", 2),
}


class GGUFAdapter(ModelAdapter):
    @classmethod
    def can_handle(cls, path: Path) -> bool:
        return path.suffix.lower() == ".gguf"

    def _parse(self):
        with open(self.path, "rb") as f:
            magic = f.read(4)
            if magic != GGUF_MAGIC:
                raise ValueError("Not a GGUF file")
            version = _read_u32(f)
            tensor_count = _read_u64(f)
            kv_count = _read_u64(f)

            kv = {}
            for _ in range(kv_count):
                key = _read_string(f)
                vtype = _read_u32(f)
                value = _read_value(f, vtype, version)
                kv[key] = (vtype, value)

            tensors = []
            for _ in range(tensor_count):
                name = _read_string(f)
                n_dims = _read_u32(f)
                shape = [_read_u64(f) for _ in range(n_dims)]
                ggml_type = _read_u32(f)
                _offset = _read_u64(f)

                type_name, bpe = GGML_TYPE_INFO.get(ggml_type, ("UNKNOWN", 4))
                param_count = 1
                for d in shape:
                    param_count *= d
                size_bytes = int(param_count * bpe)

                tensors.append(TensorInfo(
                    name=name,
                    shape=list(reversed(shape)),
                    dtype=type_name,
                    param_count=param_count,
                    size_bytes=size_bytes,
                    layer_path=self._layer_path(name),
                ))

        return kv, tensors

    def get_tensors(self) -> list[TensorInfo]:
        _, tensors = self._parse()
        return tensors

    def get_metadata(self) -> list[ModelMetadata]:
        kv, _ = self._parse()
        result = []
        for key, (vtype, value) in kv.items():
            type_name = GGUF_TYPE.get(vtype, ("unknown",))[0]
            result.append(ModelMetadata(key=key, value=value, value_type=type_name))
        return result

    def get_overview(self) -> ModelOverview:
        kv, tensors = self._parse()
        meta_map = {k: v for k, (_, v) in kv.items()}
        metadata = [
            ModelMetadata(key=k, value=v, value_type=GGUF_TYPE.get(t, ("unknown",))[0])
            for k, (t, v) in kv.items()
        ]

        total_params = sum(t.param_count for t in tensors)
        arch = meta_map.get("general.architecture")
        vocab_size = meta_map.get(f"{arch}.vocab_size") if arch else None
        quant_types = {t.dtype for t in tensors if t.dtype not in ("F32", "F16", "BF16")}
        quant = ", ".join(sorted(quant_types)) if quant_types else None

        return ModelOverview(
            id=str(self.path),
            name=meta_map.get("general.name", self.path.stem),
            path=str(self.path),
            format=ModelFormat.gguf,
            file_size=self.path.stat().st_size,
            tensor_count=len(tensors),
            param_count=total_params,
            quantization=quant,
            architecture=arch,
            vocab_size=int(vocab_size) if vocab_size is not None else None,
            metadata=metadata,
        )


def _read_u8(f): return struct.unpack("<B", f.read(1))[0]
def _read_u16(f): return struct.unpack("<H", f.read(2))[0]
def _read_u32(f): return struct.unpack("<I", f.read(4))[0]
def _read_u64(f): return struct.unpack("<Q", f.read(8))[0]
def _read_i8(f): return struct.unpack("<b", f.read(1))[0]
def _read_i16(f): return struct.unpack("<h", f.read(2))[0]
def _read_i32(f): return struct.unpack("<i", f.read(4))[0]
def _read_i64(f): return struct.unpack("<q", f.read(8))[0]
def _read_f32(f): return struct.unpack("<f", f.read(4))[0]
def _read_f64(f): return struct.unpack("<d", f.read(8))[0]


def _read_string(f) -> str:
    length = _read_u64(f)
    return f.read(length).decode("utf-8", errors="replace")


# Fixed-size element byte sizes (0 = variable length, must read individually)
_FIXED_BPE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_PREVIEW_LIMIT = 64  # max items stored for display


def _read_value(f, vtype: int, version: int = 3):
    if vtype == 0: return _read_u8(f)
    if vtype == 1: return _read_i8(f)
    if vtype == 2: return _read_u16(f)
    if vtype == 3: return _read_i16(f)
    if vtype == 4: return _read_u32(f)
    if vtype == 5: return _read_i32(f)
    if vtype == 6: return _read_f32(f)
    if vtype == 7: return bool(_read_u8(f))
    if vtype == 8: return _read_string(f)
    if vtype == 9:
        elem_type = _read_u32(f)
        count = _read_u64(f)
        bpe = _FIXED_BPE.get(elem_type, 0)

        if bpe > 0:
            # Fixed-size elements: read preview, bulk-skip the rest
            preview_count = min(count, _PREVIEW_LIMIT)
            items = [_read_value(f, elem_type, version) for _ in range(preview_count)]
            remaining = count - preview_count
            if remaining > 0:
                f.seek(remaining * bpe, 1)  # seek relative to current pos
            return items
        else:
            # Variable-length elements (strings, nested arrays): must read all
            # to keep file pointer correct; store only a preview
            items = []
            for i in range(count):
                val = _read_value(f, elem_type, version)
                if i < _PREVIEW_LIMIT:
                    items.append(val)
            return items

    if vtype == 10: return _read_u64(f)
    if vtype == 11: return _read_i64(f)
    if vtype == 12: return _read_f64(f)
    raise ValueError(f"Unknown GGUF value type: {vtype}")
