import { useState, useCallback } from 'react'
import { api } from '../../api/client'
import { useAsync } from '../../hooks/useAsync'
import { useStore } from '../../store'
import { formatBytes, formatCount } from '../../utils/format'
import { TensorDataViewer } from '../TensorDataViewer/TensorDataViewer'
import type { TensorInfo } from '../../types'
import styles from './TensorExplorer.module.css'

type SortKey = 'name' | 'param_count' | 'size_bytes' | 'dtype'

const PAGE_SIZE_OPTIONS = [
  { label: '25', value: 25 },
  { label: '50', value: 50 },
  { label: '100', value: 100 },
  { label: '250', value: 250 },
  { label: 'All', value: 99999 },
]

export function TensorExplorer({ modelId }: { modelId: string }) {
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(50)
  const [sortBy, setSortBy] = useState<SortKey>('name')
  const [sortDesc, setSortDesc] = useState(false)
  const [selectedTensor, setSelectedTensor] = useState<TensorInfo | null>(null)

  const { data, loading, error } = useAsync(
    () => api.getTensors(modelId, { page, page_size: pageSize, search, sort_by: sortBy, sort_desc: sortDesc }),
    [modelId, page, pageSize, search, sortBy, sortDesc]
  )

  const overview = useStore(s => s.overview)

  const handleSort = useCallback((key: SortKey) => {
    if (sortBy === key) setSortDesc(d => !d)
    else { setSortBy(key); setSortDesc(false) }
    setPage(1)
  }, [sortBy])

  const handleSearch = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setSearch(e.target.value)
    setPage(1)
    setSelectedTensor(null)
  }, [])

  const handlePageSize = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    setPageSize(Number(e.target.value))
    setPage(1)
  }, [])

  const handleRowClick = useCallback((t: TensorInfo) => {
    setSelectedTensor(prev => prev?.name === t.name ? null : t)
  }, [])

  const isFiltered = search.trim().length > 0
  const totalPages = data ? Math.ceil(data.total / pageSize) : 1

  const SortIcon = ({ k }: { k: SortKey }) =>
    sortBy === k ? <span className={styles.sortIcon}>{sortDesc ? '↓' : '↑'}</span> : null

  return (
    <div className={styles.outer}>
      <div className={styles.listPane}>
        <div className={styles.toolbar}>
          <input
            className={styles.search}
            placeholder="Filter tensors… (e.g. q_proj, layer.12)"
            value={search}
            onChange={handleSearch}
          />
          <div className={styles.pageSizeWrap}>
            <label className={styles.pageSizeLabel}>Show:</label>
            <select className={styles.pageSizeSelect} value={pageSize} onChange={handlePageSize}>
              {PAGE_SIZE_OPTIONS.map(o => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </div>
          {data && (
            <div className={styles.stats}>
              {isFiltered && <span className={styles.filteredBadge}>Filtered</span>}
              <span className={styles.statItem}>
                <span className={styles.statNum}>{data.total.toLocaleString()}</span>
                <span className={styles.statLbl}>tensors</span>
              </span>
              <span className={styles.statDivider}>·</span>
              <span className={styles.statItem}>
                <span className={styles.statNum}>{formatCount(data.total_params)}</span>
                <span className={styles.statLbl}>params</span>
              </span>
              <span className={styles.statDivider}>·</span>
              <span className={styles.statItem}>
                <span className={styles.statNum}>{formatBytes(data.total_size)}</span>
                <span className={styles.statLbl}>size</span>
              </span>
            </div>
          )}
        </div>

        {error && <div className={styles.error}>{error}</div>}

        <div className={styles.tableWrap}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th onClick={() => handleSort('name')} className={styles.sortable}>Name <SortIcon k="name" /></th>
                <th>Shape</th>
                <th onClick={() => handleSort('dtype')} className={styles.sortable}>Dtype <SortIcon k="dtype" /></th>
                <th onClick={() => handleSort('param_count')} className={`${styles.sortable} ${styles.right}`}>Params <SortIcon k="param_count" /></th>
                <th onClick={() => handleSort('size_bytes')} className={`${styles.sortable} ${styles.right}`}>Size <SortIcon k="size_bytes" /></th>
              </tr>
            </thead>
            <tbody>
              {loading && <tr><td colSpan={5} className={styles.loading}>Loading…</td></tr>}
              {!loading && data?.tensors.map(t => (
                <tr
                  key={t.name}
                  className={`${styles.row} ${selectedTensor?.name === t.name ? styles.rowSelected : ''}`}
                  onClick={() => handleRowClick(t)}
                >
                  <td className={styles.tensorName}>{t.name}</td>
                  <td className={styles.shape}>[{t.shape.join(', ')}]</td>
                  <td><span className={styles.dtype}>{t.dtype}</span></td>
                  <td className={styles.right}>{formatCount(t.param_count)}</td>
                  <td className={styles.right}>{formatBytes(t.size_bytes)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {data && totalPages > 1 && (
          <div className={styles.pagination}>
            <button disabled={page === 1} onClick={() => setPage(p => p - 1)}>← Prev</button>
            <span>Page {page} of {totalPages}</span>
            <button disabled={page === totalPages} onClick={() => setPage(p => p + 1)}>Next →</button>
          </div>
        )}
      </div>

      {selectedTensor && (
        <div className={styles.dataPane}>
          <div className={styles.dataPaneHeader}>
            <span className={styles.dataPaneName}>{selectedTensor.name}</span>
            <button className={styles.dataPaneClose} onClick={() => setSelectedTensor(null)}>✕</button>
          </div>
          <div className={styles.dataPaneBody}>
            <TensorDataViewer modelId={modelId} tensor={selectedTensor} />
          </div>
        </div>
      )}
    </div>
  )
}
