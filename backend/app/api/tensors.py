from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
from app.models.schemas import TensorListResponse, TensorInfo, AtlasNode
from app.state import model_cache

router = APIRouter(prefix="/api/models", tags=["tensors"])


@router.get("/{model_id}/tensors", response_model=TensorListResponse)
def list_tensors(
    model_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1),
    search: str = Query(""),
    sort_by: str = Query("name"),
    sort_desc: bool = Query(False),
):
    adapter = _get_adapter(model_id)
    tensors = adapter.get_tensors()

    if search:
        q = search.lower()
        tensors = [t for t in tensors if q in t.name.lower()]

    reverse = sort_desc
    if sort_by == "name":
        tensors.sort(key=lambda t: t.name, reverse=reverse)
    elif sort_by == "param_count":
        tensors.sort(key=lambda t: t.param_count, reverse=reverse)
    elif sort_by == "size_bytes":
        tensors.sort(key=lambda t: t.size_bytes, reverse=reverse)
    elif sort_by == "dtype":
        tensors.sort(key=lambda t: t.dtype, reverse=reverse)

    total = len(tensors)
    total_params = sum(t.param_count for t in tensors)
    total_size = sum(t.size_bytes for t in tensors)
    start = (page - 1) * page_size
    page_tensors = tensors[start: start + page_size]

    return TensorListResponse(
        tensors=page_tensors,
        total=total,
        total_params=total_params,
        total_size=total_size,
        page=page,
        page_size=page_size,
    )


@router.get("/{model_id}/atlas", response_model=AtlasNode)
def get_atlas(model_id: str):
    adapter = _get_adapter(model_id)
    tensors = adapter.get_tensors()
    return _build_tree(tensors, adapter.path.name if hasattr(adapter, 'path') else "model")


def _build_tree(tensors: list[TensorInfo], root_name: str) -> AtlasNode:
    root = AtlasNode(name=root_name, path="", size=0, param_count=0)
    nodes: dict[str, AtlasNode] = {"": root}

    for tensor in tensors:
        parts = tensor.name.split(".")
        current_path = ""
        for i, part in enumerate(parts[:-1]):
            parent_path = current_path
            current_path = f"{current_path}.{part}" if current_path else part
            if current_path not in nodes:
                node = AtlasNode(name=part, path=current_path, size=0, param_count=0)
                nodes[current_path] = node
                nodes[parent_path].children.append(node)

        leaf = AtlasNode(
            name=parts[-1],
            path=tensor.name,
            size=tensor.size_bytes,
            param_count=tensor.param_count,
            tensor=tensor,
        )
        parent_path = ".".join(parts[:-1]) if len(parts) > 1 else ""
        nodes.get(parent_path, root).children.append(leaf)

    # bubble up sizes
    _accumulate(root)
    return root


def _accumulate(node: AtlasNode) -> None:
    for child in node.children:
        _accumulate(child)
    if node.children:
        node.size = sum(c.size for c in node.children)
        node.param_count = sum(c.param_count for c in node.children)


def _get_adapter(model_id: str):
    from app.parsers.registry import get_adapter
    if model_id not in model_cache:
        path = Path(model_id)
        if path.exists():
            try:
                adapter = get_adapter(path)
                model_cache[model_id] = adapter
                return adapter
            except Exception:
                pass
        raise HTTPException(status_code=404, detail="Model not loaded")
    return model_cache[model_id]
