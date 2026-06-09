import { useEffect, useRef, useState } from 'react'
import mermaid from 'mermaid'
import { api } from '../../api/client'
import { useAsync } from '../../hooks/useAsync'
import { TensorDataViewer } from '../TensorDataViewer/TensorDataViewer'
import type { TensorInfo } from '../../types'
import styles from './GraphView.module.css'

mermaid.initialize({
  startOnLoad: false,
  theme: 'dark',
  themeVariables: {
    background:           '#0f1117',
    mainBkg:              '#161b27',
    nodeBorder:           '#1e2535',
    lineColor:            '#475569',
    primaryColor:         '#161b27',
    primaryBorderColor:   '#6366f1',
    primaryTextColor:     '#e2e8f0',
    secondaryColor:       '#1e2535',
    tertiaryColor:        '#0f1117',
    edgeLabelBackground:  '#0f1117',
    clusterBkg:           '#0f1117',
    clusterBorder:        '#334155',
    titleColor:           '#94a3b8',
    fontFamily:           'ui-monospace, monospace',
  },
  flowchart: { curve: 'basis', padding: 20 },
})

let _uid = 0

export function GraphView({ modelId }: { modelId: string }) {
  const { data, loading, error } = useAsync(() => api.getGraph(modelId), [modelId])
  const svgRef    = useRef<HTMLDivElement>(null)
  const [svgHtml, setSvgHtml]   = useState('')
  const [natural, setNatural]   = useState({ w: 0, h: 0 })
  const [zoom, setZoom]         = useState(1)
  const [selected, setSelected] = useState<TensorInfo | null>(null)

  // Render mermaid → SVG string
  useEffect(() => {
    if (!data?.diagram) return
    const id = `mermaid-${++_uid}`
    mermaid.render(id, data.diagram)
      .then(({ svg }) => {
        const vb = svg.match(/viewBox="[^"]*?\s+([0-9.]+)\s+([0-9.]+)"/)
        if (vb) {
          const w = parseFloat(vb[1])
          const h = parseFloat(vb[2])
          const fixed = svg
            .replace(/max-width\s*:\s*[^;"]+(;|(?="))/g, '')
            .replace(/<svg /, `<svg width="${w}" height="${h}" `)
          setSvgHtml(fixed)
          setNatural({ w, h })
        } else {
          setSvgHtml(svg)
        }
      })
      .catch(console.error)
  }, [data])

  // After SVG is injected into DOM, attach click handlers via addEventListener
  useEffect(() => {
    if (!svgHtml || !svgRef.current || !data?.node_map) return
    const svgEl = svgRef.current.querySelector('svg')
    if (!svgEl) return

    const nodeMap = data.node_map
    const cleanups: (() => void)[] = []

    for (const [nodeId, tensorInfo] of Object.entries(nodeMap)) {
      // Mermaid generates IDs like "flowchart-L0Q-42"
      const el = svgEl.querySelector(`[id*="-${nodeId}-"]`) ?? svgEl.querySelector(`[id="${nodeId}"]`)
      if (!el) continue

      const handler = (e: Event) => {
        e.stopPropagation()
        setSelected(tensorInfo as TensorInfo)
      }
      el.addEventListener('click', handler)
      cleanups.push(() => el.removeEventListener('click', handler))
    }

    return () => cleanups.forEach(fn => fn())
  }, [svgHtml, data])

  if (loading) return <div className={styles.center}>Building diagram…</div>
  if (error)   return <div className={styles.error}>{error}</div>

  const wrapperStyle = natural.w
    ? { width: natural.w * zoom, height: natural.h * zoom }
    : {}

  return (
    <div className={styles.outer}>
      <div className={styles.toolbar}>
        <span className={styles.label}>Model Architecture Graph</span>
        <div className={styles.zoomControls}>
          <button onClick={() => setZoom(z => Math.max(0.1, +(z - 0.1).toFixed(1)))}>−</button>
          <span className={styles.zoomVal}>{Math.round(zoom * 100)}%</span>
          <button onClick={() => setZoom(z => Math.min(4, +(z + 0.1).toFixed(1)))}>+</button>
          <button onClick={() => setZoom(1)}>Reset</button>
        </div>
      </div>

      <div className={styles.body}>
        <div className={styles.canvas}>
          <div style={wrapperStyle}>
            <div
              ref={svgRef}
              className={styles.diagram}
              style={{ transform: `scale(${zoom})`, transformOrigin: 'top left' }}
              dangerouslySetInnerHTML={{ __html: svgHtml }}
            />
          </div>
        </div>

        {selected && (
          <div className={styles.panel}>
            <div className={styles.panelHeader}>
              <span className={styles.panelTitle}>{selected.name}</span>
              <button className={styles.panelClose} onClick={() => setSelected(null)}>✕</button>
            </div>
            <div className={styles.panelContent}>
              <TensorDataViewer modelId={modelId} tensor={selected} />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
