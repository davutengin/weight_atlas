"""
Build a Mermaid flowchart diagram from a model's tensor list.
Handles both HuggingFace (q_proj / layers.N) and GGUF (attn_q / blk.N) naming.
Each layer is a subgraph with internal Q/K/V/O and Gate/Up/Down connections.
"""
import re
from app.models.schemas import TensorInfo, ModelOverview


def _fmt(n: int) -> str:
    if n >= 1_000_000_000: return f"{n/1_000_000_000:.1f}B"
    if n >= 1_000_000:     return f"{n/1_000_000:.1f}M"
    if n >= 1_000:         return f"{n/1_000:.1f}K"
    return str(n)

def _s(shape: list[int]) -> str:
    return "×".join(str(d) for d in shape)

def _find(tensors: list[TensorInfo], *kws) -> TensorInfo | None:
    for kw in kws:
        for t in tensors:
            if kw in t.name:
                return t
    return None

def _layer_count(tensors: list[TensorInfo]) -> int:
    max_idx = -1
    for t in tensors:
        m = re.search(r'(?:layers?|blocks?|blk|h)\.(\d+)\.', t.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1 if max_idx >= 0 else 0

def _is_gguf(tensors: list[TensorInfo]) -> bool:
    return any('blk.' in t.name or 'token_embd' in t.name for t in tensors)

def _pick(tensors: list[TensorInfo], prefix: str, *keys) -> TensorInfo | None:
    for t in tensors:
        if prefix in t.name:
            for k in keys:
                if k in t.name:
                    return t
    return None

def _layer_tensors(tensors, idx, gguf):
    p = f'blk.{idx}.' if gguf else f'layers.{idx}.'
    if gguf:
        return {
            'q':    _pick(tensors, p, 'attn_q'),
            'k':    _pick(tensors, p, 'attn_k'),
            'v':    _pick(tensors, p, 'attn_v'),
            'o':    _pick(tensors, p, 'attn_output'),
            'gate': _pick(tensors, p, 'ffn_gate'),
            'up':   _pick(tensors, p, 'ffn_up'),
            'down': _pick(tensors, p, 'ffn_down'),
        }
    else:
        return {
            'q':    _pick(tensors, p, 'q_proj'),
            'k':    _pick(tensors, p, 'k_proj'),
            'v':    _pick(tensors, p, 'v_proj'),
            'o':    _pick(tensors, p, 'o_proj'),
            'gate': _pick(tensors, p, 'gate_proj'),
            'up':   _pick(tensors, p, 'up_proj'),
            'down': _pick(tensors, p, 'down_proj'),
        }

def _lbl(t: TensorInfo | None, name: str) -> str:
    return f"{name}<br/>{_s(t.shape)}" if t else name


def _layer_subgraph(lt: dict, i: int) -> list[str]:
    """Emit one layer subgraph with full internal connections."""
    n = f'L{i}'   # prefix for node IDs

    lines = [f'    subgraph {n}["Layer {i}"]', '        direction TB']

    # ── Attention ──────────────────────────────────────────────────────────
    lines += [
        f'        subgraph {n}A["⚡ Attention"]',
        '            direction TB',
        f'            {n}LN1["LayerNorm"]',
        f'            {n}Q["{_lbl(lt["q"], "Q")}"]',
        f'            {n}K["{_lbl(lt["k"], "K")}"]',
        f'            {n}V["{_lbl(lt["v"], "V")}"]',
        f'            {n}AS["QKᵀ / √d"]',
        f'            {n}AV["Attn × V"]',
        f'            {n}O["{_lbl(lt["o"], "O")}"]',
        # connections
        f'            {n}LN1 --> {n}Q',
        f'            {n}LN1 --> {n}K',
        f'            {n}LN1 --> {n}V',
        f'            {n}Q --> {n}AS',
        f'            {n}K --> {n}AS',
        f'            {n}AS --> {n}AV',
        f'            {n}V --> {n}AV',
        f'            {n}AV --> {n}O',
        '        end',
    ]

    # ── MLP (SwiGLU) ───────────────────────────────────────────────────────
    lines += [
        f'        subgraph {n}M["🧮 MLP"]',
        '            direction TB',
        f'            {n}LN2["LayerNorm"]',
        f'            {n}GT["{_lbl(lt["gate"], "Gate")}"]',
        f'            {n}UP["{_lbl(lt["up"], "Up")}"]',
        f'            {n}SL["SiLU"]',
        f'            {n}MX["×"]',
        f'            {n}DW["{_lbl(lt["down"], "Down")}"]',
        # connections
        f'            {n}LN2 --> {n}GT',
        f'            {n}LN2 --> {n}UP',
        f'            {n}GT --> {n}SL',
        f'            {n}SL --> {n}MX',
        f'            {n}UP --> {n}MX',
        f'            {n}MX --> {n}DW',
        '        end',
    ]

    # Attention → MLP
    lines += [
        f'        {n}A --> {n}M',
        '    end',
    ]

    return lines


def build_mermaid(tensors: list[TensorInfo], overview: ModelOverview) -> tuple[str, dict[str, TensorInfo]]:
    gguf        = _is_gguf(tensors)
    layer_count = _layer_count(tensors)

    emb      = _find(tensors, 'token_embd', 'embed_tokens', 'wte', 'word_embeddings')
    norm_out = _find(tensors, 'output_norm', 'norm.weight', 'ln_f', 'final_layernorm')
    lm_head  = _find(tensors, 'output.weight', 'lm_head', 'embed_out')

    lines = ['flowchart LR', '']

    # Input + Embedding
    emb_lbl = f"📦 Embedding<br/>{_s(emb.shape)}<br/>{_fmt(emb.param_count)} params" if emb else "📦 Embedding"
    lines += [
        '    INPUT(["🔤 Input"])',
        f'    EMB["{emb_lbl}"]',
        '    INPUT --> EMB',
        '',
    ]

    prev = 'EMB'

    if layer_count > 0:
        lt0 = _layer_tensors(tensors, 0, gguf)

        # All layers share the same internal structure (layer 0 shapes used for all)
        for i in range(layer_count):
            lines += _layer_subgraph(lt0, i)
            lines.append(f'    {prev} --> L{i}')
            lines.append('')
            prev = f'L{i}'

    # Output
    if norm_out:
        lines.append(f'    NORM["📐 Output Norm<br/>{_s(norm_out.shape)}"]')
        lines.append(f'    {prev} --> NORM')
        prev = 'NORM'

    if lm_head:
        lines.append(f'    LM_HEAD["🎯 LM Head<br/>{_s(lm_head.shape)}<br/>{_fmt(lm_head.param_count)} params"]')
        lines.append(f'    {prev} --> LM_HEAD')
        prev = 'LM_HEAD'

    lines.append(f'    {prev} --> LOGITS(["📊 Output"])')

    # Styling
    lines += [
        '',
        '    classDef default fill:#161b27,stroke:#1e2535,color:#e2e8f0,font-size:11px',
        '    classDef io      fill:#1e2a45,stroke:#6366f1,color:#a5b4fc',
        '    class INPUT,LOGITS io',
    ]

    # Build node_map for clickable tensor nodes
    node_map: dict[str, TensorInfo] = {}
    for i in range(layer_count):
        lt = _layer_tensors(tensors, i, gguf)
        mapping = {'Q': lt['q'], 'K': lt['k'], 'V': lt['v'], 'O': lt['o'],
                   'GT': lt['gate'], 'UP': lt['up'], 'DW': lt['down']}
        for suffix, tensor in mapping.items():
            if tensor is not None:
                node_map[f'L{i}{suffix}'] = tensor

    # Highlight clickable nodes
    if node_map:
        clickable_ids = ','.join(node_map.keys())
        lines += [
            '    classDef clickable fill:#1e2a45,stroke:#6366f1,color:#a5b4fc,cursor:pointer',
            f'    class {clickable_ids} clickable',
        ]

    return '\n'.join(lines), node_map
