import { useState, useMemo, useCallback } from 'react'
import ReactECharts from 'echarts-for-react'
import { api } from '../../api/client'
import { useAsync } from '../../hooks/useAsync'
import { formatBytes, formatCount } from '../../utils/format'
import { TensorDataViewer } from '../TensorDataViewer/TensorDataViewer'
import type { AtlasNode } from '../../types'
import styles from './AtlasView.module.css'

export function AtlasView({ modelId }: { modelId: string }) {
  const { data: root, loading, error } = useAsync(() => api.getAtlas(modelId), [modelId])
  const [selectedPath, setSelectedPath] = useState('')

  const selectedNode = useMemo(() => {
    if (!root) return null
    if (!selectedPath) return root
    return findNode(root, selectedPath)
  }, [root, selectedPath])

  if (loading) return <div className={styles.center}>Building atlas…</div>
  if (error) return <div className={styles.error}>{error}</div>
  if (!root) return null

  return (
    <div className={styles.layout}>
      <aside className={styles.tree}>
        <div className={styles.treeHeader}>Model Structure</div>
        <div className={styles.treeScroll}>
          <TreeNode
            node={root}
            depth={0}
            selectedPath={selectedPath}
            onSelect={setSelectedPath}
          />
        </div>
      </aside>

      <section className={styles.detail}>
        {selectedNode && (
          selectedNode.tensor ? (
            <TensorDetail
              node={selectedNode}
              modelId={modelId}
              breadcrumb={<Breadcrumb path={selectedPath} onNavigate={setSelectedPath} rootName={root.name} />}
            />
          ) : (
            <NodeDetail
              node={selectedNode}
              onDrillDown={setSelectedPath}
              breadcrumb={<Breadcrumb path={selectedPath} onNavigate={setSelectedPath} rootName={root.name} />}
            />
          )
        )}
      </section>
    </div>
  )
}

// ── Tree Node ──────────────────────────────────────────────────────────────

function TreeNode({
  node, depth, selectedPath, onSelect,
}: {
  node: AtlasNode
  depth: number
  selectedPath: string
  onSelect: (path: string) => void
}) {
  const [open, setOpen] = useState(depth < 2)
  const hasChildren = node.children.length > 0
  const isSelected = node.path === selectedPath
  const isLeaf = !!node.tensor

  return (
    <div>
      <div
        className={`${styles.treeRow} ${isSelected ? styles.treeSelected : ''} ${isLeaf ? styles.treeLeaf : ''}`}
        style={{ paddingLeft: `${12 + depth * 16}px` }}
        onClick={() => {
          onSelect(node.path)
          if (hasChildren) setOpen(o => !o)
        }}
      >
        <span className={styles.treeToggle}>
          {hasChildren ? (open ? '▾' : '▸') : '·'}
        </span>
        <span className={styles.treeName} title={node.name}>{node.name}</span>
        {!isLeaf && node.children.length > 0 && (
          <span className={styles.treeCount}>{node.children.length}</span>
        )}
      </div>
      {open && hasChildren && (
        <div>
          {node.children.map(child => (
            <TreeNode
              key={child.path}
              node={child}
              depth={depth + 1}
              selectedPath={selectedPath}
              onSelect={onSelect}
            />
          ))}
        </div>
      )}
    </div>
  )
}

// ── Breadcrumb ─────────────────────────────────────────────────────────────

function Breadcrumb({ path, onNavigate, rootName }: {
  path: string; onNavigate: (p: string) => void; rootName: string
}) {
  const parts = path ? path.split('.') : []
  const crumbs = [{ label: rootName, path: '' }]
  parts.forEach((p, i) => {
    crumbs.push({ label: p, path: parts.slice(0, i + 1).join('.') })
  })

  return (
    <div className={styles.breadcrumb}>
      {crumbs.map((c, i) => (
        <span key={c.path}>
          {i > 0 && <span className={styles.sep}>›</span>}
          <button
            className={`${styles.crumb} ${i === crumbs.length - 1 ? styles.crumbActive : ''}`}
            onClick={() => onNavigate(c.path)}
          >
            {c.label}
          </button>
        </span>
      ))}
    </div>
  )
}

// ── Node Detail (intermediate — shows treemap of children) ─────────────────

function NodeDetail({ node, onDrillDown, breadcrumb }: { node: AtlasNode; onDrillDown: (p: string) => void; breadcrumb: React.ReactNode }) {
  const option = useMemo(() => buildTreemapOption(node), [node])

  const handleClick = useCallback((params: { data?: { path?: string } }) => {
    const path = params?.data?.path
    if (path) onDrillDown(path)
  }, [onDrillDown])

  return (
    <div className={styles.detailInner}>
      <div className={styles.nodeStats}>
        <Stat label="Children" value={String(node.children.length)} />
        <Stat label="Parameters" value={formatCount(node.param_count)} />
        <Stat label="Size" value={formatBytes(node.size)} />
      </div>
      {breadcrumb}
      <div className={styles.treemapWrap}>
        <ReactECharts
          option={option}
          style={{ height: '100%', width: '100%' }}
          theme="dark"
          opts={{ renderer: 'canvas' }}
          onEvents={{ click: handleClick }}
        />
      </div>
    </div>
  )
}

// ── Tensor Detail (leaf) ───────────────────────────────────────────────────

function TensorDetail({ node, modelId, breadcrumb }: { node: AtlasNode; modelId: string; breadcrumb: React.ReactNode }) {
  const t = node.tensor!
  return (
    <div className={styles.detailInner}>
      <div className={styles.tensorTitle}>{t.name}</div>
      <div className={styles.nodeStats}>
        <Stat label="Shape" value={`[${t.shape.join(', ')}]`} />
        <Stat label="Dtype" value={t.dtype} />
        <Stat label="Parameters" value={formatCount(t.param_count)} />
        <Stat label="Size" value={formatBytes(t.size_bytes)} />
      </div>
      {breadcrumb}
      <div className={styles.dataViewer}>
        <TensorDataViewer modelId={modelId} tensor={t} />
      </div>
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className={styles.stat}>
      <div className={styles.statLabel}>{label}</div>
      <div className={styles.statValue}>{value}</div>
    </div>
  )
}

// ── ECharts option ─────────────────────────────────────────────────────────

function countTensors(node: AtlasNode): number {
  if (node.tensor) return 1
  return node.children.reduce((s, c) => s + countTensors(c), 0)
}

function toEchartsNode(node: AtlasNode, totalSize: number): Record<string, unknown> {
  const shape = node.tensor?.shape ?? null
  const dtype = node.tensor?.dtype ?? null
  const tensorCount = countTensors(node)
  const pct = totalSize > 0 ? (node.size / totalSize) * 100 : 0
  const base = {
    name: node.name,
    value: node.size || 1,
    params: node.param_count,
    path: node.path,
    shape,
    dtype,
    isLeaf: !!node.tensor,
    tensorCount,
    pct,
  }
  if (node.children.length > 0) {
    return { ...base, children: node.children.map(c => toEchartsNode(c, totalSize)) }
  }
  return base
}

type EChartsNodeData = { name: string; value: number; params: number; path: string; shape: number[] | null; dtype: string | null; isLeaf: boolean; tensorCount: number; pct: number }

function buildTreemapOption(node: AtlasNode) {
  const totalSize = node.size || 1
  const data = node.children.map(c => toEchartsNode(c, totalSize))

  return {
    backgroundColor: 'transparent',
    tooltip: {
      formatter: (info: { name?: string; value?: number; data?: EChartsNodeData }) => {
        const d = info?.data
        if (!d || d.params == null) return info?.name ?? ''
        const size = typeof info.value === 'number' ? info.value : d.value ?? 0
        const pct = d.pct ?? 0
        const pctStr = pct >= 0.1 ? pct.toFixed(2) : '<0.1'
        const rows = [
          `<b style="color:#e2e8f0">${d.path || d.name}</b>`,
          d.shape ? `Shape &nbsp;<span style="color:#a5b4fc">[${d.shape.join(' × ')}]</span>${d.dtype ? ` &nbsp;<span style="color:#64748b">${d.dtype}</span>` : ''}` : null,
          `Tensors &nbsp;<span style="color:#a5b4fc">${d.tensorCount ?? '—'}</span>`,
          `Params &nbsp;<span style="color:#a5b4fc">${formatCount(d.params)}</span>`,
          `Size &nbsp;<span style="color:#a5b4fc">${formatBytes(size)}</span> &nbsp;<span style="color:#64748b">(${pctStr}%)</span>`,
        ].filter(Boolean).join('<br/>')
        return `<div style="font-family:monospace;font-size:12px;line-height:1.8;padding:2px 4px">${rows}</div>`
      },
    },
    series: [{
      type: 'treemap',
      data,
      width: '100%',
      height: '100%',
      roam: false,
      nodeClick: 'zoomToNode',
      leafDepth: 3,
      label: {
        show: true,
        formatter: (p: { data: EChartsNodeData }) => labelText(p.data),
        fontSize: 11,
        color: 'rgba(226,232,240,0.9)',
        overflow: 'truncate',
        lineOverflow: 'truncate',
      },
      upperLabel: {
        show: true,
        height: 22,
        fontSize: 11,
        fontWeight: 600,
        color: 'rgba(241,245,249,0.95)',
        formatter: (p: { data: EChartsNodeData }) => p.data.name,
        backgroundColor: 'rgba(0,0,0,0.35)',
      },
      itemStyle: {
        borderColor: '#0f1117',
        borderWidth: 2,
        gapWidth: 2,
      },
      levels: [
        {
          // level 0 — top groups (embed_tokens, layers, norm…)
          colorSaturation: [0.22, 0.38],
          itemStyle: { borderColor: '#0f1117', borderWidth: 4, gapWidth: 4 },
          upperLabel: { show: true, height: 22 },
          label: { fontSize: 12, fontWeight: 600 },
        },
        {
          // level 1 — sub-groups (layer.0, layer.1…)
          colorSaturation: [0.16, 0.28],
          itemStyle: { borderColorSaturation: 0.4, borderWidth: 2, gapWidth: 2 },
          upperLabel: { show: true, height: 18, fontSize: 10 },
          label: { fontSize: 10 },
        },
        {
          // level 2 — leaf tensors or deeper groups
          colorSaturation: [0.10, 0.20],
          itemStyle: { borderColorSaturation: 0.3, borderWidth: 1, gapWidth: 1 },
          upperLabel: { show: true, height: 16, fontSize: 9 },
          label: { fontSize: 9 },
        },
        {
          // level 3 — leaf tensors at deepest
          colorSaturation: [0.08, 0.16],
          itemStyle: { borderWidth: 1, gapWidth: 1 },
          label: { fontSize: 9 },
        },
      ],
      // Muted, slate-toned palette — distinct but not glaring
      color: ['#3b4f7a', '#3d5a6b', '#3a5c5a', '#4a5240', '#5a4a3a', '#5a3a4a', '#4a3a5a'],
      breadcrumb: {
        show: true,
        bottom: 4,
        height: 22,
        itemStyle: {
          color: '#161b27',
          borderColor: '#1e2535',
          shadowBlur: 0,
          textStyle: { color: '#64748b', fontSize: 11 },
        },
        emphasis: { itemStyle: { color: '#1e2535', textStyle: { color: '#94a3b8' } } },
      },
    }],
  }
}

// ── Helpers ────────────────────────────────────────────────────────────────

function labelText(d: EChartsNodeData): string {
  const pctStr = d.pct >= 0.1 ? `${d.pct.toFixed(1)}%` : '<0.1%'

  if (d.isLeaf && d.shape) {
    const shapeStr = d.shape.join('×')
    const paramStr = formatCount(d.params)
    return `${shapeStr}\n${paramStr}  ${pctStr}`
  }
  // Intermediate: name + tensor count + %
  const tStr = `${d.tensorCount} tensor${d.tensorCount !== 1 ? 's' : ''}`
  return `${d.name}\n${tStr}  ${pctStr}`
}

function findNode(node: AtlasNode, path: string): AtlasNode | null {
  if (node.path === path) return node
  for (const child of node.children) {
    const found = findNode(child, path)
    if (found) return found
  }
  return null
}
