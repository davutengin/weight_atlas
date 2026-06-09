import { useState, useCallback } from 'react'
import { api } from '../../api/client'
import { useStore } from '../../store'
import styles from './ModelLoader.module.css'

const EXAMPLES = [
  'C:\\models\\llama-3-8b.gguf',
  '/home/user/models/mistral-7b-instruct.safetensors',
  '/home/user/mlx-models/qwen2-7b/',
]

export function ModelLoader() {
  const [path, setPath] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const { setModel, setLoading: storeLoading } = useStore()

  const handleLoad = useCallback(async () => {
    if (!path.trim()) return
    setLoading(true)
    storeLoading(true)
    setError(null)
    try {
      const result = await api.loadModel(path.trim())
      setModel(result.model_id, result.overview)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load model')
    } finally {
      setLoading(false)
      storeLoading(false)
    }
  }, [path, setModel, storeLoading])

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleLoad()
  }

  return (
    <div className={styles.container}>
      <div className={styles.hero}>
        <div className={styles.icon}>⬡</div>
        <h1 className={styles.title}>Weight Atlas</h1>
        <p className={styles.subtitle}>
          Explore the architecture and weights of local AI model files.<br />
          Supports <strong>SafeTensors</strong>, <strong>GGUF</strong>, and <strong>MLX</strong> formats.
        </p>

        <div className={styles.inputGroup}>
          <input
            className={styles.input}
            placeholder="Enter path to model file or directory…"
            value={path}
            onChange={e => setPath(e.target.value)}
            onKeyDown={handleKey}
            autoFocus
          />
          <button
            className={styles.btn}
            onClick={handleLoad}
            disabled={loading || !path.trim()}
          >
            {loading ? 'Loading…' : 'Open Model'}
          </button>
        </div>

        {error && <div className={styles.error}>{error}</div>}

        <div className={styles.examples}>
          <span className={styles.examplesLabel}>Examples:</span>
          {EXAMPLES.map(ex => (
            <button key={ex} className={styles.exBtn} onClick={() => setPath(ex)}>
              {ex}
            </button>
          ))}
        </div>
      </div>

      <div className={styles.features}>
        {[
          { icon: '🗺️', label: 'Atlas View', desc: 'Interactive treemap of model structure' },
          { icon: '🔍', label: 'Tensor Explorer', desc: 'Search and sort all tensors' },
          { icon: '📊', label: 'Model Overview', desc: 'Parameters, quantization, architecture' },
          { icon: '🏷️', label: 'Metadata', desc: 'Embedded model metadata' },
        ].map(f => (
          <div key={f.label} className={styles.feature}>
            <div className={styles.featureIcon}>{f.icon}</div>
            <div className={styles.featureLabel}>{f.label}</div>
            <div className={styles.featureDesc}>{f.desc}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
