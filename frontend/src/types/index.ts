export type ModelFormat = 'safetensors' | 'gguf' | 'mlx' | 'unknown'

export interface TensorInfo {
  name: string
  shape: number[]
  dtype: string
  param_count: number
  size_bytes: number
  layer_path: string
}

export interface ModelMetadata {
  key: string
  value: unknown
  value_type: string
}

export interface ModelOverview {
  id: string
  name: string
  path: string
  format: ModelFormat
  file_size: number
  tensor_count: number
  param_count: number
  quantization: string | null
  architecture: string | null
  vocab_size: number | null
  metadata: ModelMetadata[]
}

export interface TensorListResponse {
  tensors: TensorInfo[]
  total: number
  total_params: number
  total_size: number
  page: number
  page_size: number
}

export interface AtlasNode {
  name: string
  path: string
  size: number
  param_count: number
  children: AtlasNode[]
  tensor: TensorInfo | null
}

export interface TensorDataResponse {
  name: string
  shape: number[]
  dtype: string
  total_rows: number
  total_cols: number
  row_offset: number
  col_offset: number
  row_count: number
  col_count: number
  data: number[][]
  stats: { min: number; max: number; mean: number; std: number }
}

export interface SearchResult {
  tensors: TensorInfo[]
  query: string
  total: number
}

export interface LoadModelResponse {
  model_id: string
  overview: ModelOverview
}
