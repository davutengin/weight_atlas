"""
Trainability analysis — LoRA target detection, architecture fingerprinting,
preset generation and fine-tuning readiness scoring.
"""
import re
from collections import defaultdict
from typing import Optional
from app.models.schemas import (
    TensorInfo, ModelOverview,
    TrainabilityAnalysis, LoRACandidate, FineTuningPreset,
    ArchitectureFingerprint, QLoRAAnalysis, ReadinessScore,
)

LORA_RANK = 16  # default rank for adapter size estimation

# Attention projections — primary LoRA targets
# Covers HuggingFace naming (q_proj…) AND GGUF/llama.cpp naming (attn_q…)
CORE_MODULES = [
    # HuggingFace / SafeTensors
    'q_proj', 'k_proj', 'v_proj', 'o_proj',
    'query', 'key', 'value', 'out_proj',
    'query_key_value', 'c_attn', 'c_proj',
    'wq', 'wk', 'wv', 'wo',
    # GGUF / llama.cpp
    'attn_q', 'attn_k', 'attn_v', 'attn_output',
    'attn_qkv', 'attn_kv',
]

# MLP projections — optional / extended targets
OPTIONAL_MODULES = [
    # HuggingFace / SafeTensors
    'gate_proj', 'up_proj', 'down_proj',
    'fc1', 'fc2',
    'dense', 'dense_h_to_4h', 'dense_4h_to_h',
    'fc_in', 'fc_out', 'w1', 'w2', 'w3',
    # GGUF / llama.cpp
    'ffn_gate', 'ffn_up', 'ffn_down',
    'ffn_gate_inp', 'ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps',
]

ARCH_KEYWORDS = {
    'Qwen':      ['qwen'],
    'LLaMA':     ['llama'],
    'Mistral':   ['mistral'],
    'Phi':       ['phi'],
    'Gemma':     ['gemma'],
    'Falcon':    ['falcon'],
    'GPT-NeoX':  ['gpt_neox', 'neox'],
    'GPT-2':     ['gpt2'],
    'BERT':      ['bert'],
    'Mamba':     ['mamba'],
    'DeepSeek':  ['deepseek'],
    'StarCoder': ['starcoder'],
    'Baichuan':  ['baichuan'],
    'InternLM':  ['internlm'],
    'Yi':        ['yi'],
    'ChatGLM':   ['chatglm', 'glm'],
}


def analyze(tensors: list[TensorInfo], overview: ModelOverview) -> TrainabilityAnalysis:
    # 1 — Scan tensors for module patterns
    module_tensors: dict[str, list[TensorInfo]] = defaultdict(list)
    all_modules = set(CORE_MODULES + OPTIONAL_MODULES)

    for t in tensors:
        parts = re.split(r'[./]', t.name)
        for part in parts:
            if part in all_modules:
                module_tensors[part].append(t)

    found_core     = [m for m in CORE_MODULES     if m in module_tensors]
    found_optional = [m for m in OPTIONAL_MODULES if m in module_tensors]

    # 2 — LoRA candidates
    lora_candidates: list[LoRACandidate] = []
    for m in found_core:
        ts = module_tensors[m]
        lora_candidates.append(LoRACandidate(
            module=m, layer_count=len(ts),
            param_count=sum(t.param_count for t in ts), priority='core',
        ))
    for m in found_optional:
        ts = module_tensors[m]
        lora_candidates.append(LoRACandidate(
            module=m, layer_count=len(ts),
            param_count=sum(t.param_count for t in ts), priority='optional',
        ))

    # 3 — Architecture fingerprint
    fingerprint = _fingerprint(tensors, overview, found_core)

    # 4 — Presets
    presets = _build_presets(module_tensors, found_core, found_optional, overview.param_count)

    # 5 — QLoRA analysis
    qlora = _qlora_analysis(overview, fingerprint)

    # 6 — Readiness score
    readiness = _readiness(found_core, found_optional, fingerprint, overview)

    # 7 — Recommended modules & preset
    priority_order = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                      'query', 'key', 'value', 'out_proj', 'c_attn']
    recommended_modules = [m for m in priority_order if m in found_core][:4]
    if not recommended_modules:
        recommended_modules = found_core[:4] or found_optional[:4]

    if len(recommended_modules) >= 4:
        recommended_preset = 'Standard LoRA'
    elif recommended_modules:
        recommended_preset = 'Minimal LoRA'
    else:
        recommended_preset = 'Full Fine-Tuning'

    # 8 — Summary
    summary = _summary(fingerprint, qlora, recommended_preset, recommended_modules, presets)

    return TrainabilityAnalysis(
        fingerprint=fingerprint,
        lora_candidates=lora_candidates,
        presets=presets,
        qlora=qlora,
        readiness=readiness,
        summary=summary,
        recommended_preset=recommended_preset,
        recommended_modules=recommended_modules,
    )


# ── Helpers ────────────────────────────────────────────────────────────────

def _lora_params(tensors: list[TensorInfo], rank: int = LORA_RANK) -> int:
    total = 0
    for t in tensors:
        if len(t.shape) >= 2:
            out_f, in_f = t.shape[0], t.shape[1]
            total += rank * (in_f + out_f)
    return total


def _build_presets(
    module_tensors: dict[str, list[TensorInfo]],
    found_core: list[str],
    found_optional: list[str],
    total_params: int,
) -> list[FineTuningPreset]:

    def preset(name: str, want: list[str]) -> Optional[FineTuningPreset]:
        available = [m for m in want if m in module_tensors]
        if not available:
            return None
        params = sum(_lora_params(module_tensors[m]) for m in available)
        size_mb = round((params * 2) / (1024 ** 2), 2)
        pct = round((params / total_params * 100) if total_params else 0, 3)
        return FineTuningPreset(name=name, modules=available,
                                trainable_params=params, adapter_size_mb=size_mb,
                                pct_of_model=pct)

    attn_2  = [m for m in ['q_proj', 'v_proj', 'query', 'value'] if m in module_tensors][:2]
    attn_4  = [m for m in ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                            'query', 'key', 'value', 'out_proj'] if m in module_tensors][:4]
    mlp     = [m for m in ['gate_proj', 'up_proj', 'down_proj',
                            'fc1', 'fc2', 'w1', 'w2', 'w3'] if m in module_tensors]
    extended = attn_4 + mlp

    results = []
    for name, mods in [('Minimal LoRA', attn_2),
                        ('Standard LoRA', attn_4),
                        ('Extended LoRA', extended)]:
        p = preset(name, mods)
        if p:
            results.append(p)
    return results


def _fingerprint(
    tensors: list[TensorInfo],
    overview: ModelOverview,
    found_core: list[str],
) -> ArchitectureFingerprint:

    arch_meta  = (overview.architecture or '').lower()
    all_names  = ' '.join(t.name for t in tensors).lower()
    reasons    = []
    best_fam   = 'Unknown'
    best_score = 0

    for fam, kws in ARCH_KEYWORDS.items():
        score = 0
        for kw in kws:
            if kw in arch_meta:
                score += 60
            if kw in all_names:
                score += 25
        if score > best_score:
            best_score = score
            best_fam   = fam

    if arch_meta:
        reasons.append(f'Architecture metadata: {overview.architecture}')
    if found_core:
        reasons.append(f'Standard attention projections: {", ".join(found_core[:4])}')

    layer_count     = _count_layers(tensors)
    hidden_size     = _hidden_size(tensors)
    attention_heads = _attn_heads(overview)

    if layer_count:
        reasons.append(f'{layer_count} transformer layers')
    if hidden_size:
        reasons.append(f'Hidden size: {hidden_size}')
    if overview.vocab_size:
        reasons.append(f'Vocabulary size: {overview.vocab_size:,}')

    confidence = min(99, best_score)
    if best_fam == 'Unknown' and found_core:
        best_fam   = 'Transformer (unidentified)'
        confidence = 40
        reasons.append('Standard transformer structure detected, family unidentified')

    return ArchitectureFingerprint(
        family=best_fam, confidence=confidence, reasons=reasons[:5],
        hidden_size=hidden_size, layer_count=layer_count, attention_heads=attention_heads,
    )


def _count_layers(tensors: list[TensorInfo]) -> Optional[int]:
    max_idx = -1
    for t in tensors:
        # HuggingFace: layers.0, blocks.0, h.0
        # GGUF/llama.cpp: blk.0
        m = re.search(r'(?:layers?|blocks?|blk|h)\.(\d+)\.', t.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1 if max_idx >= 0 else None


def _hidden_size(tensors: list[TensorInfo]) -> Optional[int]:
    # HuggingFace: embed_tokens, wte, word_embeddings
    # GGUF: token_embd
    for t in tensors:
        if any(kw in t.name for kw in ('embed_tokens', 'wte', 'word_embeddings', 'token_embd')):
            if len(t.shape) >= 2:
                return t.shape[1]
    # Fallback: q_proj or attn_q weight
    for t in tensors:
        if any(kw in t.name for kw in ('q_proj', 'attn_q')) and len(t.shape) >= 2:
            return t.shape[1]
    return None


def _attn_heads(overview: ModelOverview) -> Optional[int]:
    for m in overview.metadata:
        if any(k in m.key for k in ('attention_head_count', 'num_attention_heads', 'num_heads')):
            try:
                return int(m.value)
            except Exception:
                pass
    return None


def _qlora_analysis(overview: ModelOverview, fp: ArchitectureFingerprint) -> QLoRAAnalysis:
    reasons   = []
    is_quant  = bool(overview.quantization)

    if is_quant:
        reasons.append(f'Model is already quantized ({overview.quantization})')
        quant_status = f'Quantized ({overview.quantization})'
        strategy = 'QLoRA'
    else:
        reasons.append('Full-precision model — can be quantized on-the-fly for QLoRA')
        quant_status = 'Not quantized (full precision)'
        strategy = 'QLoRA'

    if fp.family not in ('Unknown', 'BERT'):
        reasons.append('Decoder-only transformer architecture')

    if fp.layer_count and fp.layer_count > 4:
        reasons.append(f'{fp.layer_count} transformer layers suitable for LoRA')

    compatible = fp.family not in ('Unknown',) or is_quant

    return QLoRAAnalysis(
        compatible=compatible,
        quantization_status=quant_status,
        recommended_strategy=strategy,
        reasons=reasons,
    )


def _readiness(
    found_core: list[str],
    found_optional: list[str],
    fp: ArchitectureFingerprint,
    overview: ModelOverview,
) -> ReadinessScore:
    score   = 0
    reasons = []

    if found_core:
        score += 35
        reasons.append('Standard attention projection modules detected')
    if fp.family not in ('Unknown',):
        score += 25
        reasons.append(f'Recognized architecture family ({fp.family})')
    if fp.layer_count and 4 <= fp.layer_count <= 200:
        score += 15
        reasons.append(f'Layer count within typical fine-tuning range ({fp.layer_count})')
    if overview.metadata:
        score += 10
        reasons.append('Model metadata available')
    if overview.quantization:
        score += 10
        reasons.append(f'Quantization detected — QLoRA compatible')
    if found_optional:
        score += 5
        reasons.append('MLP projection modules available for extended LoRA')

    return ReadinessScore(score=min(score, 100), reasons=reasons)


def _fmt(n: int) -> str:
    if n >= 1_000_000_000: return f'{n/1_000_000_000:.1f}B'
    if n >= 1_000_000:     return f'{n/1_000_000:.1f}M'
    if n >= 1_000:         return f'{n/1_000:.1f}K'
    return str(n)


def _summary(
    fp: ArchitectureFingerprint,
    qlora: QLoRAAnalysis,
    preset: str,
    modules: list[str],
    presets: list[FineTuningPreset],
) -> str:
    param_str = next(
        (_fmt(p.trainable_params) for p in presets if p.name == preset), 'N/A'
    )
    qlora_str = 'High' if qlora.compatible else 'Low'
    complexity = 'Low' if len(modules) <= 4 else 'Medium'
    fam_desc = (
        f'This model appears to be a {fp.family}-family decoder-only transformer.'
        if fp.family not in ('Unknown', 'Transformer (unidentified)')
        else 'Architecture family could not be conclusively determined.'
    )
    parts = [
        fam_desc,
        f'Recommended fine-tuning strategy: {preset}.',
        (f'Recommended target modules: {", ".join(modules)}.' if modules else ''),
        f'Estimated trainable parameters: {param_str}.',
        f'QLoRA compatibility: {qlora_str}.',
        f'Complexity: {complexity}.',
    ]
    return ' '.join(p for p in parts if p)
