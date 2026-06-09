import type { ModelOverview } from '../../types'
import { formatBytes, formatCount } from '../../utils/format'
import styles from './ModelOverview.module.css'

export function ModelOverviewPanel({ overview }: { overview: ModelOverview }) {
  const stats = [
    { label: 'Format', value: overview.format.toUpperCase() },
    { label: 'File Size', value: formatBytes(overview.file_size) },
    { label: 'Tensors', value: overview.tensor_count.toLocaleString() },
    { label: 'Parameters', value: formatCount(overview.param_count) },
    { label: 'Architecture', value: overview.architecture ?? '—' },
    { label: 'Quantization', value: overview.quantization ?? 'None (full precision)' },
    { label: 'Vocab Size', value: overview.vocab_size?.toLocaleString() ?? '—' },
  ]

  return (
    <div className={styles.container}>
      <h1 className={styles.title}>{overview.name}</h1>
      <p className={styles.path}>{overview.path}</p>

      <div className={styles.grid}>
        {stats.map(s => (
          <div key={s.label} className={styles.stat}>
            <div className={styles.statLabel}>{s.label}</div>
            <div className={styles.statValue}>{s.value}</div>
          </div>
        ))}
      </div>

      {overview.metadata.length > 0 && (
        <section className={styles.section}>
          <h2 className={styles.sectionTitle}>Embedded Metadata</h2>
          <div className={styles.metaList}>
            {overview.metadata.slice(0, 20).map(m => (
              <div key={m.key} className={styles.metaRow}>
                <span className={styles.metaKey}>{m.key}</span>
                <span className={styles.metaVal}>{String(m.value)}</span>
              </div>
            ))}
            {overview.metadata.length > 20 && (
              <div className={styles.metaMore}>+{overview.metadata.length - 20} more — see Metadata tab</div>
            )}
          </div>
        </section>
      )}
    </div>
  )
}
