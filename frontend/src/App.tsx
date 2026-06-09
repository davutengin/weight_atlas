import { useStore } from './store'
import { Shell } from './components/Layout/Shell'
import { ModelLoader } from './components/ModelLoader/ModelLoader'
import { ModelOverviewPanel } from './components/ModelOverview/ModelOverview'
import { AtlasView } from './components/AtlasView/AtlasView'
import { TensorExplorer } from './components/TensorExplorer/TensorExplorer'
import { MetadataExplorer } from './components/MetadataExplorer/MetadataExplorer'

export default function App() {
  const { modelId, overview, activeTab } = useStore()

  const content = () => {
    if (!modelId || !overview) return <ModelLoader />
    switch (activeTab) {
      case 'overview': return <div style={{ position: 'absolute', inset: 0, overflowY: 'auto' }}><ModelOverviewPanel overview={overview} /></div>
      case 'atlas': return <AtlasView modelId={modelId} />
      case 'tensors': return <TensorExplorer modelId={modelId} />
      case 'metadata': return <MetadataExplorer overview={overview} />
    }
  }

  return <Shell>{content()}</Shell>
}
