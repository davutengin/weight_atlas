import { useStore } from '../../store'
import styles from './Shell.module.css'

const TABS = [
  { id: 'overview',      label: 'Overview' },
  { id: 'atlas',         label: 'Atlas' },
  { id: 'tensors',       label: 'Tensors' },
  { id: 'metadata',      label: 'Metadata' },
  { id: 'trainability',  label: 'Trainability' },
] as const

export function Shell({ children }: { children: React.ReactNode }) {
  const { overview, activeTab, setTab, clear } = useStore()

  return (
    <div className={styles.shell}>
      <header className={styles.header}>
        <div className={styles.brand}>
          <span className={styles.logo}>⬡</span>
          <span className={styles.name}>Weight Atlas</span>
        </div>
        {overview && (
          <nav className={styles.tabs}>
            {TABS.map(t => (
              <button
                key={t.id}
                className={`${styles.tab} ${activeTab === t.id ? styles.active : ''}`}
                onClick={() => setTab(t.id)}
              >
                {t.label}
              </button>
            ))}
          </nav>
        )}
        {overview && (
          <div className={styles.modelBadge}>
            <span className={styles.modelName}>{overview.name}</span>
            <span className={styles.formatBadge}>{overview.format.toUpperCase()}</span>
            <button className={styles.closeBtn} onClick={clear} title="Close model">✕</button>
          </div>
        )}
      </header>
      <main className={styles.main}>{children}</main>
    </div>
  )
}
