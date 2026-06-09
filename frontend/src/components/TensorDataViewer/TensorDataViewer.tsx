import { useState, useMemo, useCallback, useRef, useEffect } from 'react'
import ReactECharts from 'echarts-for-react'
import { api } from '../../api/client'
import { useAsync } from '../../hooks/useAsync'
import { formatCount } from '../../utils/format'
import type { TensorInfo, TensorDataResponse } from '../../types'
import styles from './TensorDataViewer.module.css'

const PAGE = 64

type ViewMode = 'chart' | 'table'

interface Props {
  modelId: string
  tensor: TensorInfo
}

export function TensorDataViewer({ modelId, tensor }: Props) {
  const is1D = tensor.shape.length <= 1
  const totalRows = is1D ? 1 : tensor.shape[0]
  const totalCols = is1D ? (tensor.shape[0] ?? 1) : tensor.shape.slice(1).reduce((a, b) => a * b, 1)

  const [viewMode, setViewMode] = useState<ViewMode>('chart')
  const [rowOffset, setRowOffset] = useState(0)
  const [colOffset, setColOffset] = useState(0)

  const rowCount = Math.min(PAGE, totalRows - rowOffset)
  const colCount = Math.min(PAGE, totalCols - colOffset)

  const { data, loading, error } = useAsync(
    () => api.getTensorData(modelId, tensor.name, { row_offset: rowOffset, row_count: rowCount, col_offset: colOffset, col_count: colCount }),
    [modelId, tensor.name, rowOffset, colOffset]
  )

  const chartOption = useMemo(() => {
    if (!data) return {}
    if (is1D || data.row_count === 1) return build1DOption(data.data[0] ?? [], data.col_offset)
    return build2DOption(data)
  }, [data, is1D])

  const rowPages = Math.ceil(totalRows / PAGE)
  const colPages = Math.ceil(totalCols / PAGE)
  const rowPage = Math.floor(rowOffset / PAGE)
  const colPage = Math.floor(colOffset / PAGE)

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.shapeInfo}>
          <span className={styles.badge}>{tensor.dtype}</span>
          <span className={styles.dim}>shape [{tensor.shape.join(' × ')}]</span>
          <span className={styles.dim}>{formatCount(tensor.param_count)} params</span>
        </div>
        <div className={styles.headerRight}>
          {data && (
            <div className={styles.statsRow}>
              {Object.entries(data.stats).map(([k, v]) => (
                <span key={k} className={styles.stat}>
                  <span className={styles.statK}>{k}</span>
                  <span className={styles.statV}>{Number(v).toFixed(5)}</span>
                </span>
              ))}
            </div>
          )}
          <div className={styles.viewToggle}>
            <button className={`${styles.toggleBtn} ${viewMode === 'chart' ? styles.toggleActive : ''}`} onClick={() => setViewMode('chart')}>Chart</button>
            <button className={`${styles.toggleBtn} ${viewMode === 'table' ? styles.toggleActive : ''}`} onClick={() => setViewMode('table')}>Table</button>
          </div>
        </div>
      </div>

      <div className={styles.chartWrap}>
        {loading && <div className={styles.overlay}>Loading data…</div>}
        {error && <div className={styles.overlayErr}>{error}</div>}

        {viewMode === 'chart' && data && !loading && (
          <ReactECharts option={chartOption} style={{ height: '100%', width: '100%' }} theme="dark" opts={{ renderer: 'canvas' }} />
        )}

        {viewMode === 'table' && (
          <VirtualTable
            modelId={modelId}
            tensorName={tensor.name}
            totalRows={totalRows}
            totalCols={totalCols}
            initialData={data ?? null}
          />
        )}
      </div>

      {/* Chart-mode pagination only */}
      {viewMode === 'chart' && (
        <div className={styles.nav}>
          {rowPages > 1 && (
            <div className={styles.navGroup}>
              <span className={styles.navLabel}>Rows</span>
              <button disabled={rowPage === 0} onClick={() => setRowOffset(0)}>⏮</button>
              <button disabled={rowPage === 0} onClick={() => setRowOffset(o => Math.max(0, o - PAGE))}>‹</button>
              <span className={styles.navPage}>{rowOffset}–{Math.min(rowOffset + PAGE, totalRows) - 1}<span className={styles.navTotal}> / {totalRows}</span></span>
              <button disabled={rowOffset + PAGE >= totalRows} onClick={() => setRowOffset(o => o + PAGE)}>›</button>
              <button disabled={rowOffset + PAGE >= totalRows} onClick={() => setRowOffset((rowPages - 1) * PAGE)}>⏭</button>
            </div>
          )}
          {colPages > 1 && (
            <div className={styles.navGroup}>
              <span className={styles.navLabel}>Cols</span>
              <button disabled={colPage === 0} onClick={() => setColOffset(0)}>⏮</button>
              <button disabled={colPage === 0} onClick={() => setColOffset(o => Math.max(0, o - PAGE))}>‹</button>
              <span className={styles.navPage}>{colOffset}–{Math.min(colOffset + PAGE, totalCols) - 1}<span className={styles.navTotal}> / {totalCols}</span></span>
              <button disabled={colOffset + PAGE >= totalCols} onClick={() => setColOffset(o => o + PAGE)}>›</button>
              <button disabled={colOffset + PAGE >= totalCols} onClick={() => setColOffset((colPages - 1) * PAGE)}>⏭</button>
            </div>
          )}
          {rowPages === 1 && colPages === 1 && data && (
            <span className={styles.navLabel}>All {data.row_count * data.col_count} values shown</span>
          )}
        </div>
      )}
    </div>
  )
}

// ── Infinite-scroll table ─────────────────────────────────────────────────
// Renders only the current PAGE×PAGE window as a plain sticky table.
// When the user scrolls near an edge, the next page is fetched and the
// scroll position is nudged so navigation feels seamless.

const EDGE = 80   // px from rendered edge that triggers a page fetch

interface VirtualTableProps {
  modelId: string
  tensorName: string
  totalRows: number
  totalCols: number
  initialData: TensorDataResponse | null
}

function VirtualTable({ modelId, tensorName, totalRows, totalCols, initialData }: VirtualTableProps) {
  const wrapRef   = useRef<HTMLDivElement>(null)
  const tableRef  = useRef<HTMLTableElement>(null)
  const busy      = useRef(false)
  const skipScroll = useRef(false)

  const [rowOffset, setRowOffset] = useState(0)
  const [colOffset, setColOffset] = useState(0)
  const [data, setData]   = useState<TensorDataResponse | null>(initialData)
  const [fetching, setFetching] = useState(false)

  // Fetch whenever offsets change
  useEffect(() => {
    if (busy.current) return
    busy.current = true
    setFetching(true)
    const rc = Math.min(PAGE, totalRows - rowOffset)
    const cc = Math.min(PAGE, totalCols - colOffset)
    api.getTensorData(modelId, tensorName, {
      row_offset: rowOffset, row_count: rc,
      col_offset: colOffset, col_count: cc,
    })
      .then(d => { setData(d) })
      .catch(() => {})
      .finally(() => { setFetching(false); busy.current = false })
  }, [modelId, tensorName, rowOffset, colOffset])

  // After a page change: nudge scroll to the appropriate edge so the
  // user can keep scrolling in the same direction.
  const prevOffsets = useRef({ row: 0, col: 0 })
  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const prev = prevOffsets.current
    skipScroll.current = true

    if (colOffset > prev.col) {
      // moved right → place scroll near left edge of new content
      el.scrollLeft = EDGE + 10
    } else if (colOffset < prev.col) {
      // moved left → place scroll near right edge of new content
      el.scrollLeft = el.scrollWidth - el.clientWidth - EDGE - 10
    }
    if (rowOffset > prev.row) {
      el.scrollTop = EDGE + 10
    } else if (rowOffset < prev.row) {
      el.scrollTop = el.scrollHeight - el.clientHeight - EDGE - 10
    }

    prevOffsets.current = { row: rowOffset, col: colOffset }
    requestAnimationFrame(() => { skipScroll.current = false })
  }, [rowOffset, colOffset])

  const handleScroll = useCallback(() => {
    if (skipScroll.current || busy.current) return
    const el = wrapRef.current
    if (!el) return
    const { scrollLeft, scrollTop, scrollWidth, scrollHeight, clientWidth, clientHeight } = el

    const nearRight  = scrollLeft + clientWidth  >= scrollWidth  - EDGE
    const nearLeft   = scrollLeft <= EDGE
    const nearBottom = scrollTop  + clientHeight >= scrollHeight - EDGE
    const nearTop    = scrollTop  <= EDGE

    let nextCol = colOffset
    let nextRow = rowOffset

    if      (nearRight  && colOffset + PAGE < totalCols) nextCol = colOffset + PAGE
    else if (nearLeft   && colOffset > 0)                nextCol = Math.max(0, colOffset - PAGE)

    if      (nearBottom && rowOffset + PAGE < totalRows) nextRow = rowOffset + PAGE
    else if (nearTop    && rowOffset > 0)                nextRow = Math.max(0, rowOffset - PAGE)

    if (nextCol !== colOffset) setColOffset(nextCol)
    if (nextRow !== rowOffset) setRowOffset(nextRow)
  }, [colOffset, rowOffset, totalCols, totalRows])

  const stats = data?.stats ?? { min: -1, max: 1 }
  const range = (stats.max - stats.min) || 1

  const cellBg = useCallback((v: number) => {
    const t = (v - stats.min) / range
    return t < 0.5
      ? `rgba(37,99,235,${(0.5 - t) * 0.8})`
      : `rgba(220,38,38,${(t - 0.5) * 0.8})`
  }, [stats.min, range])

  return (
    <div className={styles.tableScrollWrap} ref={wrapRef} onScroll={handleScroll}>
      {fetching && <div className={styles.fetchingBar}>Loading…</div>}
      {data && (
        <table className={styles.stickyTable} ref={tableRef}>
          <thead>
            <tr>
              <th className={styles.cornerTh} />
              {Array.from({ length: data.col_count }, (_, i) => (
                <th key={i} className={styles.colTh}>{colOffset + i}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.data.map((row, ri) => (
              <tr key={ri}>
                <td className={styles.rowTh}>{rowOffset + ri}</td>
                {row.map((v, ci) => (
                  <td
                    key={ci}
                    className={styles.dataTd}
                    style={{ background: cellBg(v) }}
                    title={`[${rowOffset + ri}, ${colOffset + ci}] = ${v}`}
                  >
                    {formatNum(v)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

// ── Helpers ────────────────────────────────────────────────────────────────

function formatNum(v: number): string {
  if (!isFinite(v)) return String(v)
  const abs = Math.abs(v)
  if (abs === 0) return '0'
  if (abs >= 100) return v.toFixed(2)
  if (abs >= 1) return v.toFixed(4)
  if (abs >= 0.001) return v.toFixed(6)
  return v.toExponential(3)
}

function build1DOption(values: number[], colOffset: number) {
  return {
    backgroundColor: 'transparent',
    grid: { top: 24, right: 16, bottom: 40, left: 60 },
    tooltip: {
      trigger: 'axis',
      formatter: (p: { dataIndex: number; value: number }[]) =>
        `[${colOffset + p[0].dataIndex}] = ${p[0].value.toFixed(6)}`,
    },
    xAxis: {
      type: 'category',
      data: values.map((_, i) => colOffset + i),
      axisLabel: { color: '#475569', fontSize: 11 },
      axisLine: { lineStyle: { color: '#1e2535' } },
    },
    yAxis: {
      type: 'value',
      axisLabel: { color: '#475569', fontSize: 11 },
      splitLine: { lineStyle: { color: '#1e2535' } },
    },
    series: [{ type: 'bar', data: values, itemStyle: { color: '#6366f1' }, emphasis: { itemStyle: { color: '#818cf8' } } }],
  }
}

function build2DOption(data: { data: number[][]; row_offset: number; col_offset: number; row_count: number; col_count: number; stats: { min: number; max: number } }) {
  const { min, max } = data.stats
  const heatData: [number, number, number][] = []
  for (let r = 0; r < data.row_count; r++)
    for (let c = 0; c < data.col_count; c++)
      heatData.push([c, r, data.data[r]?.[c] ?? 0])

  return {
    backgroundColor: 'transparent',
    tooltip: {
      formatter: (p: { data: [number, number, number] }) =>
        `[${data.row_offset + p.data[1]}, ${data.col_offset + p.data[0]}] = ${p.data[2].toFixed(6)}`,
    },
    grid: { top: 10, right: 80, bottom: 40, left: 50 },
    xAxis: {
      type: 'category',
      data: Array.from({ length: data.col_count }, (_, i) => data.col_offset + i),
      axisLabel: { color: '#475569', fontSize: 10, interval: Math.floor(data.col_count / 8) },
      axisLine: { lineStyle: { color: '#1e2535' } },
      splitLine: { show: false },
    },
    yAxis: {
      type: 'category',
      data: Array.from({ length: data.row_count }, (_, i) => data.row_offset + i),
      axisLabel: { color: '#475569', fontSize: 10, interval: Math.floor(data.row_count / 8) },
      axisLine: { lineStyle: { color: '#1e2535' } },
      splitLine: { show: false },
    },
    visualMap: {
      min, max, calculable: true, orient: 'vertical', right: 0, top: 'center',
      textStyle: { color: '#475569', fontSize: 10 },
      inRange: { color: ['#1e3a5f', '#2563eb', '#ffffff', '#dc2626', '#7f1d1d'] },
    },
    series: [{ type: 'heatmap', data: heatData, emphasis: { itemStyle: { borderColor: '#6366f1', borderWidth: 1 } } }],
  }
}
