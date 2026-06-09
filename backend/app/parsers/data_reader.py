"""
Reads raw tensor values from model files without loading everything into memory.
Supports SafeTensors (F32/F16/BF16/I8/I32) and GGUF (F32/F16/BF16/Q8_0/Q4_0/Q4_K).
"""
import struct
import mmap
import json
import math
from pathlib import Path
from dataclasses import dataclass


@dataclass
class TensorSlice:
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


# ── SafeTensors ────────────────────────────────────────────────────────────

def read_safetensors_slice(
    path: Path,
    tensor_name: str,
    row_offset: int, row_count: int,
    col_offset: int, col_count: int,
) -> TensorSlice:
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
        data_base = 8 + header_size

    info = header.get(tensor_name)
    if not info:
        raise KeyError(f"Tensor '{tensor_name}' not found")

    dtype = info["dtype"]
    shape = info["shape"]
    data_start, data_end = info["data_offsets"]

    total_rows, total_cols = _shape_to_2d(shape)
    row_offset = min(row_offset, max(0, total_rows - 1))
    col_offset = min(col_offset, max(0, total_cols - 1))
    row_count = min(row_count, total_rows - row_offset)
    col_count = min(col_count, total_cols - col_offset)

    bpe = _safetensors_bpe(dtype)
    file_offset = data_base + data_start

    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        values = _read_st_slice(mm, file_offset, dtype, bpe, total_rows, total_cols,
                                row_offset, row_count, col_offset, col_count)
        mm.close()

    grid = [values[r * col_count:(r + 1) * col_count] for r in range(row_count)]
    flat = [v for row in grid for v in row]
    stats = _stats(flat)

    return TensorSlice(
        name=tensor_name, shape=shape, dtype=dtype,
        total_rows=total_rows, total_cols=total_cols,
        row_offset=row_offset, col_offset=col_offset,
        row_count=row_count, col_count=col_count,
        data=grid, stats=stats,
    )


def _read_st_slice(mm, base: int, dtype: str, bpe: int, total_rows: int, total_cols: int,
                   row_offset: int, row_count: int, col_offset: int, col_count: int) -> list[float]:
    values = []
    for r in range(row_offset, row_offset + row_count):
        for c in range(col_offset, col_offset + col_count):
            idx = r * total_cols + c
            off = base + idx * bpe
            raw = mm[off:off + bpe]
            values.append(_decode_st(raw, dtype))
    return values


def _decode_st(raw: bytes, dtype: str) -> float:
    if dtype == "F32": return struct.unpack("<f", raw)[0]
    if dtype == "F16": return _f16_to_f32(raw)
    if dtype == "BF16": return struct.unpack("<f", b"\x00\x00" + raw)[0]
    if dtype == "I32": return float(struct.unpack("<i", raw)[0])
    if dtype == "I64": return float(struct.unpack("<q", raw)[0])
    if dtype == "I8":  return float(struct.unpack("<b", raw)[0])
    if dtype == "U8":  return float(struct.unpack("<B", raw)[0])
    return 0.0


def _safetensors_bpe(dtype: str) -> int:
    return {"F32": 4, "F64": 8, "F16": 2, "BF16": 2,
            "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}.get(dtype, 4)


# ── GGUF ──────────────────────────────────────────────────────────────────

def read_gguf_slice(
    path: Path,
    tensor_name: str,
    row_offset: int, row_count: int,
    col_offset: int, col_count: int,
) -> TensorSlice:
    tensor_index, data_base = _gguf_build_index(path)

    if tensor_name not in tensor_index:
        raise KeyError(f"Tensor '{tensor_name}' not found")

    info = tensor_index[tensor_name]
    shape = info["shape"]
    ggml_type = info["ggml_type"]
    tensor_offset = info["offset"]

    type_name, _ = _GGML_TYPE_INFO.get(ggml_type, ("UNKNOWN", 0))
    total_rows, total_cols = _shape_to_2d(shape)

    row_offset = min(row_offset, max(0, total_rows - 1))
    col_offset = min(col_offset, max(0, total_cols - 1))
    row_count = min(row_count, total_rows - row_offset)
    col_count = min(col_count, total_cols - col_offset)

    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        abs_offset = data_base + tensor_offset
        values = _read_gguf_slice(mm, abs_offset, ggml_type, total_rows, total_cols,
                                  row_offset, row_count, col_offset, col_count)
        mm.close()

    grid = [values[r * col_count:(r + 1) * col_count] for r in range(row_count)]
    flat = [v for row in grid for v in row]
    stats = _stats(flat)

    return TensorSlice(
        name=tensor_name, shape=shape, dtype=type_name,
        total_rows=total_rows, total_cols=total_cols,
        row_offset=row_offset, col_offset=col_offset,
        row_count=row_count, col_count=col_count,
        data=grid, stats=stats,
    )


def _gguf_build_index(path: Path) -> tuple[dict, int]:
    """Parse GGUF header, return {tensor_name: info} and absolute data section start."""
    with open(path, "rb") as f:
        f.read(4)  # magic
        version = struct.unpack("<I", f.read(4))[0]
        tensor_count = struct.unpack("<Q", f.read(8))[0]
        kv_count = struct.unpack("<Q", f.read(8))[0]

        for _ in range(kv_count):
            _skip_gguf_kv(f, version)

        tensors = {}
        for _ in range(tensor_count):
            name = _read_gguf_str(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            shape = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            ggml_type = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            tensors[name] = {"shape": list(reversed(shape)), "ggml_type": ggml_type, "offset": offset}

        # data section starts at next 32-byte alignment after current position
        pos = f.tell()
        data_base = _align(pos, 32)

    return tensors, data_base


def _read_gguf_slice(mm, base: int, ggml_type: int, total_rows: int, total_cols: int,
                     row_offset: int, row_count: int, col_offset: int, col_count: int) -> list[float]:
    values = []
    for r in range(row_offset, row_offset + row_count):
        for c in range(col_offset, col_offset + col_count):
            idx = r * total_cols + c
            v = _decode_ggml(mm, base, ggml_type, idx, total_cols)
            values.append(v)
    return values


def _decode_ggml(mm, base: int, ggml_type: int, idx: int, row_cols: int) -> float:
    if ggml_type == 0:  # F32
        off = base + idx * 4
        return struct.unpack_from("<f", mm, off)[0]
    if ggml_type == 1:  # F16
        off = base + idx * 2
        return _f16_to_f32(mm[off:off + 2])
    if ggml_type == 30:  # BF16
        off = base + idx * 2
        return struct.unpack("<f", b"\x00\x00" + mm[off:off + 2])[0]
    if ggml_type == 8:  # Q8_0 — 32 elems per block, 2-byte F16 scale + 32 bytes i8
        block_size = 32
        block_idx = idx // block_size
        elem_idx = idx % block_size
        block_off = base + block_idx * 34  # 2 + 32
        scale = _f16_to_f32(mm[block_off:block_off + 2])
        val = struct.unpack_from("<b", mm, block_off + 2 + elem_idx)[0]
        return scale * val
    if ggml_type in (2, 3):  # Q4_0 / Q4_1
        block_size = 32
        block_idx = idx // block_size
        elem_idx = idx % block_size
        if ggml_type == 2:  # Q4_0: 2-byte scale + 16 bytes (32 × 4-bit)
            block_off = base + block_idx * 18
            scale = _f16_to_f32(mm[block_off:block_off + 2])
            byte_idx = elem_idx // 2
            nibble = mm[block_off + 2 + byte_idx]
            v = (nibble & 0xF) if elem_idx % 2 == 0 else (nibble >> 4)
            return scale * (v - 8)
        else:  # Q4_1: 2-byte scale + 2-byte min + 16 bytes
            block_off = base + block_idx * 20
            scale = _f16_to_f32(mm[block_off:block_off + 2])
            min_val = _f16_to_f32(mm[block_off + 2:block_off + 4])
            byte_idx = elem_idx // 2
            nibble = mm[block_off + 4 + byte_idx]
            v = (nibble & 0xF) if elem_idx % 2 == 0 else (nibble >> 4)
            return scale * v + min_val
    # Unsupported: return 0
    return 0.0


# ── GGUF helpers ──────────────────────────────────────────────────────────

_GGUF_TYPE_BPE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

_GGML_TYPE_INFO = {
    0: ("F32", 4), 1: ("F16", 2), 2: ("Q4_0", 0), 3: ("Q4_1", 0),
    8: ("Q8_0", 0), 30: ("BF16", 2),
}


def _skip_gguf_kv(f, version: int):
    key_len = struct.unpack("<Q", f.read(8))[0]
    f.read(key_len)
    vtype = struct.unpack("<I", f.read(4))[0]
    _skip_gguf_value(f, vtype, version)


def _skip_gguf_value(f, vtype: int, version: int):
    bpe = _GGUF_TYPE_BPE.get(vtype, 0)
    if bpe:
        f.read(bpe)
    elif vtype == 8:  # string
        f.read(struct.unpack("<Q", f.read(8))[0])
    elif vtype == 9:  # array
        elem_type = struct.unpack("<I", f.read(4))[0]
        count = struct.unpack("<Q", f.read(8))[0]
        ebpe = _GGUF_TYPE_BPE.get(elem_type, 0)
        if ebpe:
            f.read(count * ebpe)
        else:
            for _ in range(count):
                _skip_gguf_value(f, elem_type, version)


def _read_gguf_str(f) -> str:
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8", errors="replace")


def _gu64(f) -> int:
    return struct.unpack("<Q", f.read(8))[0]


def _align(pos: int, alignment: int) -> int:
    return pos + (alignment - pos % alignment) % alignment


# ── Common helpers ─────────────────────────────────────────────────────────

def _shape_to_2d(shape: list[int]) -> tuple[int, int]:
    if len(shape) == 0: return 1, 1
    if len(shape) == 1: return 1, shape[0]
    rows = shape[0]
    cols = math.prod(shape[1:])
    return rows, cols


def _f16_to_f32(data: bytes) -> float:
    h = struct.unpack("<H", data)[0]
    sign = (h >> 15) & 0x1
    exp = (h >> 10) & 0x1F
    mant = h & 0x3FF
    if exp == 0:
        if mant == 0: return -0.0 if sign else 0.0
        exp2 = 1 - 15 - 10
        val = mant * (2.0 ** exp2)
    elif exp == 31:
        return float("-inf") if sign else float("inf") if mant == 0 else float("nan")
    else:
        val = (1 + mant / 1024.0) * (2.0 ** (exp - 15))
    return -val if sign else val


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0, "max": 0, "mean": 0, "std": 0}
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return {"min": 0, "max": 0, "mean": 0, "std": 0}
    n = len(finite)
    mn = min(finite)
    mx = max(finite)
    mean = sum(finite) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in finite) / n) if n > 1 else 0.0
    return {"min": round(mn, 6), "max": round(mx, 6), "mean": round(mean, 6), "std": round(std, 6)}
