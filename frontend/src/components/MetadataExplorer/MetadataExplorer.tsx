import { useState } from 'react'
import type { ModelOverview } from '../../types'
import styles from './MetadataExplorer.module.css'

export function MetadataExplorer({ overview }: { overview: ModelOverview }) {
  const [search, setSearch] = useState('')
  const items = overview.metadata.filter(
    m => !search || m.key.toLowerCase().includes(search.toLowerCase()) ||
         String(m.value).toLowerCase().includes(search.toLowerCase())
  )

  return (
    <div className={styles.container}>
      <div className={styles.toolbar}>
        <input
          className={styles.search}
          placeholder="Filter metadata…"
          value={search}
          onChange={e => setSearch(e.target.value)}
        />
        <span className={styles.count}>{items.length} entries</span>
      </div>
      <div className={styles.tableWrap}>
        {items.length === 0 ? (
          <div className={styles.empty}>No metadata found.</div>
        ) : (
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Key</th>
                <th>Value</th>
                <th>Type</th>
              </tr>
            </thead>
            <tbody>
              {items.map(m => (
                <tr key={m.key} className={styles.row}>
                  <td className={styles.key}>{m.key}</td>
                  <td className={styles.value}>{renderValue(m.value)}</td>
                  <td className={styles.type}>{m.value_type}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}

function renderValue(v: unknown): string {
  if (Array.isArray(v)) return `[${v.slice(0, 8).join(', ')}${v.length > 8 ? ', …' : ''}]`
  if (typeof v === 'object' && v !== null) return JSON.stringify(v).slice(0, 120)
  return String(v)
}
