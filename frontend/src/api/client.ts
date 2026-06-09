import type {
  LoadModelResponse, ModelOverview, TensorListResponse,
  AtlasNode, SearchResult, TensorDataResponse, TrainabilityAnalysis,
} from '../types'

const BASE = '/api'

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(BASE + url, options)
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Request failed')
  }
  return res.json()
}

export const api = {
  loadModel: (path: string): Promise<LoadModelResponse> =>
    request('/models/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path }),
    }),

  getOverview: (modelId: string): Promise<ModelOverview> =>
    request(`/models/${encodeURIComponent(modelId)}/overview`),

  getTensors: (
    modelId: string,
    params: { page?: number; page_size?: number; search?: string; sort_by?: string; sort_desc?: boolean }
  ): Promise<TensorListResponse> => {
    const q = new URLSearchParams()
    if (params.page) q.set('page', String(params.page))
    if (params.page_size) q.set('page_size', String(params.page_size))
    if (params.search) q.set('search', params.search)
    if (params.sort_by) q.set('sort_by', params.sort_by)
    if (params.sort_desc != null) q.set('sort_desc', String(params.sort_desc))
    return request(`/models/${encodeURIComponent(modelId)}/tensors?${q}`)
  },

  getAtlas: (modelId: string): Promise<AtlasNode> =>
    request(`/models/${encodeURIComponent(modelId)}/atlas`),

  search: (modelId: string, q: string): Promise<SearchResult> =>
    request(`/models/${encodeURIComponent(modelId)}/search?q=${encodeURIComponent(q)}`),

  getTrainability: (modelId: string): Promise<TrainabilityAnalysis> =>
    request(`/models/${encodeURIComponent(modelId)}/trainability`),

  getGraph: (modelId: string): Promise<{ diagram: string; node_map: Record<string, import('../types').TensorInfo> }> =>
    request(`/models/${encodeURIComponent(modelId)}/graph`),

  getVocab: (modelId: string, offset: number, limit: number, search?: string): Promise<{ tokens: string[]; ids?: number[]; total: number; offset: number; limit: number }> => {
    const q = new URLSearchParams({ offset: String(offset), limit: String(limit) })
    if (search) q.set('search', search)
    return request(`/models/${encodeURIComponent(modelId)}/vocab?${q}`)
  },

  getTensorData: (
    modelId: string,
    name: string,
    params: { row_offset?: number; row_count?: number; col_offset?: number; col_count?: number }
  ): Promise<TensorDataResponse> => {
    const q = new URLSearchParams({ name })
    if (params.row_offset != null) q.set('row_offset', String(params.row_offset))
    if (params.row_count != null) q.set('row_count', String(params.row_count))
    if (params.col_offset != null) q.set('col_offset', String(params.col_offset))
    if (params.col_count != null) q.set('col_count', String(params.col_count))
    return request(`/models/${encodeURIComponent(modelId)}/tensor-data?${q}`)
  },
}
