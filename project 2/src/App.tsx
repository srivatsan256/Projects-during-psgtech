import React from 'react';
import { useTheme } from './hooks/useTheme';
import { useBlockchain } from './hooks/useBlockchain';
import { NetworkVisualization } from './components/NetworkVisualization';
import { BlockchainExplorer } from './components/BlockchainExplorer';
import { TransactionPanel } from './components/TransactionPanel';
import { NetworkStats } from './components/NetworkStats';
import { Blocks, Sun, Moon } from 'lucide-react';

function App() {
  const { theme, toggleTheme } = useTheme();
  const {
    nodes,
    selectedNode,
    setSelectedNode,
    networkStats,
    createTransaction,
    startMining,
    toggleNodeStatus,
    syncAllNodes,
    getSelectedNodeData
  } = useBlockchain();

  const selectedNodeData = getSelectedNodeData();

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors duration-200">
      <header className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 transition-colors duration-200">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-center relative">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-blue-600 dark:bg-blue-500 rounded-xl flex items-center justify-center transition-colors duration-200">
                <Blocks className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white transition-colors duration-200">
                  Blockchain P2P Network
                </h1>
                <p className="text-sm text-gray-500 dark:text-gray-400 transition-colors duration-200">
                  Interactive blockchain simulation with peer-to-peer consensus
                </p>
              </div>
            </div>
            
            <button
              onClick={toggleTheme}
              className="absolute right-0 p-2 rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 transition-all duration-200"
              aria-label="Toggle theme"
            >
              {theme === 'light' ? (
                <Moon className="w-5 h-5 text-gray-600 dark:text-gray-300" />
              ) : (
                <Sun className="w-5 h-5 text-gray-600 dark:text-gray-300" />
              )}
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          <NetworkVisualization
            nodes={nodes}
            selectedNode={selectedNode}
            onNodeSelect={setSelectedNode}
            onToggleNode={toggleNodeStatus}
            onStartMining={startMining}
          />
          
          <NetworkStats
            stats={networkStats}
            onSyncNodes={syncAllNodes}
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <BlockchainExplorer selectedNode={selectedNodeData} />
          
          <TransactionPanel
            nodes={nodes}
            selectedNode={selectedNode}
            onCreateTransaction={createTransaction}
          />
        </div>
      </main>
    </div>
  );
}

export default App;