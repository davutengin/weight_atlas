from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from app.api.models import _get_cached

router = APIRouter(prefix="/api/models", tags=["vocab"])


class VocabResponse(BaseModel):
    tokens: list[str]
    total: int
    offset: int
    limit: int


@router.get("/{model_id}/vocab", response_model=VocabResponse)
def get_vocab(
    model_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(500, ge=1, le=2000),
    search: str = Query(""),
):
    adapter = _get_cached(model_id)
    try:
        all_tokens = adapter.get_vocab()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if search:
        q = search.lower()
        # include token text matches AND numeric ID matches
        matched = [
            (i, t) for i, t in enumerate(all_tokens)
            if q in t.lower() or q == str(i)
        ]
        total  = len(matched)
        page   = matched[offset: offset + limit]
        tokens = [t for _, t in page]
        # Return global IDs encoded as "id:token" so frontend can show real IDs
        # Instead, we return tokens list + a parallel ids list
        ids    = [i for i, _ in page]
        return VocabSearchResponse(tokens=tokens, ids=ids, total=total, offset=offset, limit=limit)

    total = len(all_tokens)
    page  = all_tokens[offset: offset + limit]
    return VocabResponse(tokens=page, total=total, offset=offset, limit=limit)


class VocabSearchResponse(VocabResponse):
    ids: list[int] = []
