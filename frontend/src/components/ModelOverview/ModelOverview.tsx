import { useState, useEffect, useCallback } from 'react'
import type { ModelOverview } from '../../types'
import { formatBytes, formatCount } from '../../utils/format'
import { useStore } from '../../store'
import { api } from '../../api/client'
import styles from './ModelOverview.module.css'

export function ModelOverviewPanel({ overview }: { overview: ModelOverview }) {
  const [showVocab, setShowVocab] = useState(false)
  const setTab = useStore(s => s.setTab)

  const stats = [
    { label: 'Format',        value: overview.format.toUpperCase(), onClick: undefined },
    { label: 'File Size',     value: formatBytes(overview.file_size), onClick: undefined },
    { label: 'Tensors',       value: overview.tensor_count.toLocaleString(), onClick: () => setTab('tensors') },
    { label: 'Parameters',    value: formatCount(overview.param_count), onClick: undefined },
    { label: 'Architecture',  value: overview.architecture ?? '—', onClick: undefined },
    { label: 'Quantization',  value: overview.quantization ?? 'None (full precision)', onClick: undefined },
    {
      label: 'Vocab Size',
      value: overview.vocab_size?.toLocaleString() ?? '—',
      onClick: () => setShowVocab(true),
    },
  ]

  return (
    <div className={styles.container}>
      <h1 className={styles.title}>{overview.name}</h1>
      <p className={styles.path}>{overview.path}</p>

      <div className={styles.grid}>
        {stats.map(s => (
          <div
            key={s.label}
            className={`${styles.stat} ${s.onClick ? styles.statClickable : ''}`}
            onClick={s.onClick}
          >
            <div className={styles.statLabel}>
              {s.label}
              {s.onClick && <span className={styles.statHint}>{s.label === 'Tensors' ? 'click to explore' : 'click to browse'}</span>}
            </div>
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

      {showVocab && (
        <VocabModal
          modelId={overview.id}
          vocabSize={overview.vocab_size}
          onClose={() => setShowVocab(false)}
        />
      )}
    </div>
  )
}

// ── Vocab Modal ────────────────────────────────────────────────────────────

const PAGE_SIZE = 500

function VocabModal({ modelId, vocabSize, onClose }: {
  modelId: string
  vocabSize?: number | null
  onClose: () => void
}) {
  const [search, setSearch]         = useState('')
  const [debouncedSearch, setDebounced] = useState('')
  const [offset, setOffset]         = useState(0)
  const [tokens, setTokens]         = useState<string[]>([])
  const [ids, setIds]               = useState<number[] | null>(null)
  const [total, setTotal]           = useState(vocabSize ?? 0)
  const [loading, setLoading]       = useState(false)
  const [error, setError]           = useState('')

  // Debounce search input 300ms
  useEffect(() => {
    const t = setTimeout(() => setDebounced(search), 300)
    return () => clearTimeout(t)
  }, [search])

  const fetchPage = useCallback((off: number, q: string) => {
    setLoading(true)
    setError('')
    api.getVocab(modelId, off, PAGE_SIZE, q || undefined)
      .then(r => {
        setTokens(r.tokens)
        setIds(r.ids ?? null)
        setTotal(r.total)
        setOffset(off)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [modelId])

  // Refetch when debounced search or offset changes
  useEffect(() => { fetchPage(0, debouncedSearch) }, [debouncedSearch, fetchPage])

  const totalPages  = Math.ceil(total / PAGE_SIZE)
  const currentPage = Math.floor(offset / PAGE_SIZE)

  return (
    <div className={styles.modalOverlay} onClick={onClose}>
      <div className={styles.modal} onClick={e => e.stopPropagation()}>
        <div className={styles.modalHeader}>
          <div>
            <div className={styles.modalTitle}>Vocabulary</div>
            <div className={styles.modalSub}>
              {total.toLocaleString()} {debouncedSearch ? 'matches' : 'tokens total'}
              {!debouncedSearch && ` · showing ${offset + 1}–${Math.min(offset + PAGE_SIZE, total)}`}
            </div>
          </div>
          <button className={styles.modalClose} onClick={onClose}>✕</button>
        </div>

        <div className={styles.modalSearch}>
          <input
            className={styles.searchInput}
            placeholder="Search all tokens…"
            value={search}
            onChange={e => setSearch(e.target.value)}
            autoFocus
          />
          {totalPages > 1 && (
            <div className={styles.pagNav}>
              <button disabled={currentPage === 0 || loading} onClick={() => fetchPage(0, debouncedSearch)}>⏮</button>
              <button disabled={currentPage === 0 || loading} onClick={() => fetchPage(offset - PAGE_SIZE, debouncedSearch)}>‹</button>
              <span className={styles.pagInfo}>{currentPage + 1} / {totalPages}</span>
              <button disabled={currentPage >= totalPages - 1 || loading} onClick={() => fetchPage(offset + PAGE_SIZE, debouncedSearch)}>›</button>
              <button disabled={currentPage >= totalPages - 1 || loading} onClick={() => fetchPage((totalPages - 1) * PAGE_SIZE, debouncedSearch)}>⏭</button>
            </div>
          )}
        </div>

        <div className={styles.tokenGrid}>
          {loading && <div style={{ gridColumn: '1/-1', padding: '32px', textAlign: 'center', color: '#475569' }}>Loading…</div>}
          {error   && <div style={{ gridColumn: '1/-1', padding: '32px', textAlign: 'center', color: '#f87171' }}>{error}</div>}
          {!loading && !error && tokens.length === 0 && (
            <div style={{ gridColumn: '1/-1', padding: '32px', textAlign: 'center', color: '#475569', fontSize: '13px' }}>
              {debouncedSearch ? `No tokens match "${debouncedSearch}".` : 'Token data not available for this model.'}
            </div>
          )}
          {!loading && tokens.map((token, i) => {
            const realId = ids ? ids[i] : offset + i
            return (
              <div key={realId} className={styles.tokenCard} title={`ID: ${realId}`}>
                <span className={styles.tokenText}>{token === '' ? <em className={styles.tokenEmpty}>empty</em> : token}</span>
                <span className={styles.tokenId}>{realId}</span>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
