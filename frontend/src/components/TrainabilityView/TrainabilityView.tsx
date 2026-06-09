import { api } from '../../api/client'
import { useAsync } from '../../hooks/useAsync'
import { formatCount } from '../../utils/format'
import type { TrainabilityAnalysis, FineTuningPreset } from '../../types'
import styles from './TrainabilityView.module.css'

export function TrainabilityView({ modelId }: { modelId: string }) {
  const { data, loading, error } = useAsync(() => api.getTrainability(modelId), [modelId])

  if (loading) return <div className={styles.center}>Analyzing model…</div>
  if (error)   return <div className={styles.error}>{error}</div>
  if (!data)   return null

  return (
    <div style={{ position: 'absolute', inset: 0, overflowY: 'auto' }}>
      <div className={styles.container}>
        <div className={styles.topRow}>
          <FingerprintCard fp={data.fingerprint} />
          <ReadinessCard   score={data.readiness} />
          <QLoRACard       qlora={data.qlora} />
        </div>

        <div className={styles.midRow}>
          <LoRACandidatesCard candidates={data.lora_candidates} />
          <PresetsCard presets={data.presets} recommended={data.recommended_preset} />
        </div>

        <SummaryCard data={data} />
      </div>
    </div>
  )
}

// ── Architecture Fingerprint ───────────────────────────────────────────────

function FingerprintCard({ fp }: { fp: TrainabilityAnalysis['fingerprint'] }) {
  return (
    <div className={styles.card}>
      <div className={styles.cardTitle}>Architecture Fingerprint</div>
      <div className={styles.familyRow}>
        <span className={styles.familyName}>{fp.family}</span>
        <span className={styles.confidenceBadge}>{fp.confidence}% confidence</span>
      </div>
      <div className={styles.fpStats}>
        {fp.layer_count  != null && <Chip label="Layers"  value={String(fp.layer_count)} />}
        {fp.hidden_size  != null && <Chip label="Hidden"  value={String(fp.hidden_size)} />}
        {fp.attention_heads != null && <Chip label="Heads" value={String(fp.attention_heads)} />}
      </div>
      <ul className={styles.reasonList}>
        {fp.reasons.map((r, i) => <li key={i} className={styles.reason}><span className={styles.check}>✓</span>{r}</li>)}
      </ul>
    </div>
  )
}

// ── Readiness Score ────────────────────────────────────────────────────────

function ReadinessCard({ score }: { score: TrainabilityAnalysis['readiness'] }) {
  const s = score.score
  const color = s >= 80 ? '#22c55e' : s >= 50 ? '#f59e0b' : '#ef4444'
  const dash  = 2 * Math.PI * 42
  const fill  = dash * (s / 100)

  return (
    <div className={styles.card}>
      <div className={styles.cardTitle}>Fine-Tuning Readiness</div>
      <div className={styles.scoreWrap}>
        <svg width="100" height="100" viewBox="0 0 100 100">
          <circle cx="50" cy="50" r="42" fill="none" stroke="#1e2535" strokeWidth="8" />
          <circle
            cx="50" cy="50" r="42" fill="none"
            stroke={color} strokeWidth="8"
            strokeDasharray={`${fill} ${dash - fill}`}
            strokeLinecap="round"
            transform="rotate(-90 50 50)"
            style={{ transition: 'stroke-dasharray 0.6s ease' }}
          />
          <text x="50" y="46" textAnchor="middle" fill={color} fontSize="22" fontWeight="700">{s}</text>
          <text x="50" y="62" textAnchor="middle" fill="#475569" fontSize="11">/ 100</text>
        </svg>
      </div>
      <ul className={styles.reasonList}>
        {score.reasons.map((r, i) => <li key={i} className={styles.reason}><span className={styles.check}>✓</span>{r}</li>)}
      </ul>
    </div>
  )
}

// ── QLoRA Compatibility ────────────────────────────────────────────────────

function QLoRACard({ qlora }: { qlora: TrainabilityAnalysis['qlora'] }) {
  return (
    <div className={styles.card}>
      <div className={styles.cardTitle}>QLoRA Compatibility</div>
      <div className={styles.strategyRow}>
        <span className={styles.strategyLabel}>Recommended</span>
        <span className={styles.strategyBadge}>{qlora.recommended_strategy}</span>
      </div>
      <div className={`${styles.compatBadge} ${qlora.compatible ? styles.compatOk : styles.compatNo}`}>
        {qlora.compatible ? '✓ Compatible' : '✗ Incompatible'}
      </div>
      <div className={styles.quantStatus}>{qlora.quantization_status}</div>
      <ul className={styles.reasonList}>
        {qlora.reasons.map((r, i) => <li key={i} className={styles.reason}><span className={styles.bullet}>·</span>{r}</li>)}
      </ul>
    </div>
  )
}

// ── LoRA Candidates ────────────────────────────────────────────────────────

function LoRACandidatesCard({ candidates }: { candidates: TrainabilityAnalysis['lora_candidates'] }) {
  const core     = candidates.filter(c => c.priority === 'core')
  const optional = candidates.filter(c => c.priority === 'optional')

  return (
    <div className={styles.card}>
      <div className={styles.cardTitle}>LoRA Candidates</div>

      {core.length > 0 && (
        <>
          <div className={styles.groupLabel}>Core (Attention)</div>
          {core.map(c => (
            <div key={c.module} className={styles.candidateRow}>
              <span className={styles.checkGreen}>✓</span>
              <span className={styles.moduleName}>{c.module}</span>
              <span className={styles.layerCount}>{c.layer_count} layers</span>
              <span className={styles.paramHint}>{formatCount(c.param_count)}</span>
            </div>
          ))}
        </>
      )}

      {optional.length > 0 && (
        <>
          <div className={styles.groupLabel} style={{ marginTop: 16 }}>Optional (MLP)</div>
          {optional.map(c => (
            <div key={c.module} className={styles.candidateRow}>
              <span className={styles.checkOptional}>✓</span>
              <span className={styles.moduleName}>{c.module}</span>
              <span className={styles.layerCount}>{c.layer_count} layers</span>
              <span className={styles.paramHint}>{formatCount(c.param_count)}</span>
            </div>
          ))}
        </>
      )}

      {candidates.length === 0 && (
        <div className={styles.empty}>No standard LoRA targets detected.</div>
      )}
    </div>
  )
}

// ── Fine-Tuning Presets ────────────────────────────────────────────────────

function PresetsCard({ presets, recommended }: { presets: FineTuningPreset[]; recommended: string }) {
  return (
    <div className={styles.card}>
      <div className={styles.cardTitle}>Fine-Tuning Presets</div>
      <div className={styles.presetList}>
        {presets.map(p => (
          <div key={p.name} className={`${styles.preset} ${p.name === recommended ? styles.presetRecommended : ''}`}>
            <div className={styles.presetHeader}>
              <span className={styles.presetName}>{p.name}</span>
              {p.name === recommended && <span className={styles.recommendedBadge}>Recommended</span>}
            </div>
            <div className={styles.presetModules}>
              {p.modules.map(m => <span key={m} className={styles.moduleTag}>{m}</span>)}
            </div>
            <div className={styles.presetStats}>
              <div className={styles.presetStat}>
                <span className={styles.presetStatLabel}>Trainable params</span>
                <span className={styles.presetStatValue}>{formatCount(p.trainable_params)}</span>
              </div>
              <div className={styles.presetStat}>
                <span className={styles.presetStatLabel}>Adapter size</span>
                <span className={styles.presetStatValue}>{p.adapter_size_mb} MB</span>
              </div>
              <div className={styles.presetStat}>
                <span className={styles.presetStatLabel}>% of model</span>
                <span className={styles.presetStatValue}>{p.pct_of_model.toFixed(2)}%</span>
              </div>
            </div>
          </div>
        ))}
        {presets.length === 0 && <div className={styles.empty}>No presets available.</div>}
      </div>
    </div>
  )
}

// ── Summary ────────────────────────────────────────────────────────────────

function SummaryCard({ data }: { data: TrainabilityAnalysis }) {
  const preset = data.presets.find(p => p.name === data.recommended_preset)
  return (
    <div className={`${styles.card} ${styles.summaryCard}`}>
      <div className={styles.cardTitle}>Fine-Tuning Summary</div>
      <p className={styles.summaryText}>{data.summary}</p>
      <div className={styles.summaryGrid}>
        <SummaryItem label="Detected Family"      value={data.fingerprint.family} />
        <SummaryItem label="Recommended Strategy" value={data.recommended_preset} />
        <SummaryItem label="Target Modules"       value={data.recommended_modules.join(', ') || '—'} />
        <SummaryItem label="Trainable Parameters" value={preset ? formatCount(preset.trainable_params) : '—'} />
        <SummaryItem label="QLoRA Compatibility"  value={data.qlora.compatible ? 'High' : 'Low'} />
        <SummaryItem label="Complexity"           value={data.recommended_modules.length <= 4 ? 'Low' : 'Medium'} />
      </div>
    </div>
  )
}

function SummaryItem({ label, value }: { label: string; value: string }) {
  return (
    <div className={styles.summaryItem}>
      <div className={styles.summaryLabel}>{label}</div>
      <div className={styles.summaryValue}>{value}</div>
    </div>
  )
}

function Chip({ label, value }: { label: string; value: string }) {
  return (
    <div className={styles.chip}>
      <span className={styles.chipLabel}>{label}</span>
      <span className={styles.chipValue}>{value}</span>
    </div>
  )
}
