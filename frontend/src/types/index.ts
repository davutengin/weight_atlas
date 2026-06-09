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

export interface LoRACandidate {
  module: string
  layer_count: number
  param_count: number
  priority: 'core' | 'optional'
}

export interface FineTuningPreset {
  name: string
  modules: string[]
  trainable_params: number
  adapter_size_mb: number
  pct_of_model: number
}

export interface ArchitectureFingerprint {
  family: string
  confidence: number
  reasons: string[]
  hidden_size: number | null
  layer_count: number | null
  attention_heads: number | null
}

export interface QLoRAAnalysis {
  compatible: boolean
  quantization_status: string
  recommended_strategy: string
  reasons: string[]
}

export interface ReadinessScore {
  score: number
  reasons: string[]
}

export interface TrainabilityAnalysis {
  fingerprint: ArchitectureFingerprint
  lora_candidates: LoRACandidate[]
  presets: FineTuningPreset[]
  qlora: QLoRAAnalysis
  readiness: ReadinessScore
  summary: string
  recommended_preset: string
  recommended_modules: string[]
}

export interface LoadModelResponse {
  model_id: string
  overview: ModelOverview
}
