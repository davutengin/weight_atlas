import { create } from 'zustand'
import type { ModelOverview } from '../types'

type Tab = 'overview' | 'atlas' | 'tensors' | 'metadata' | 'trainability'

interface AppState {
  modelId: string | null
  overview: ModelOverview | null
  activeTab: Tab
  isLoading: boolean
  error: string | null
  setModel: (id: string, overview: ModelOverview) => void
  setTab: (tab: Tab) => void
  setLoading: (v: boolean) => void
  setError: (e: string | null) => void
  clear: () => void
}

export const useStore = create<AppState>((set) => ({
  modelId: null,
  overview: null,
  activeTab: 'overview',
  isLoading: false,
  error: null,
  setModel: (id, overview) => set({ modelId: id, overview, error: null }),
  setTab: (tab) => set({ activeTab: tab }),
  setLoading: (v) => set({ isLoading: v }),
  setError: (e) => set({ error: e }),
  clear: () => set({ modelId: null, overview: null, error: null, activeTab: 'overview' }),
}))
