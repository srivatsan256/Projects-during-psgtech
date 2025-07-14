import React from 'react';
import { Node } from '../types/blockchain';
import { Circle, Zap, Power, PowerOff } from 'lucide-react';

interface NetworkVisualizationProps {
  nodes: Node[];
  selectedNode: string | null;
  onNodeSelect: (nodeId: string) => void;
  onToggleNode: (nodeId: string) => void;
  onStartMining: (nodeId: string) => void;
}

export const NetworkVisualization: React.FC<NetworkVisualizationProps> = ({
  nodes,
  selectedNode,
  onNodeSelect,
  onToggleNode,
  onStartMining
}) => {
  const renderConnections = () => {
    return nodes.map((node, index) => 
      nodes.slice(index + 1).map((targetNode) => (
        <line
          key={`${node.id}-${targetNode.id}`}
          x1={node.position.x}
          y1={node.position.y}
          x2={targetNode.position.x}
          y2={targetNode.position.y}
          stroke={node.isOnline && targetNode.isOnline ? '#10b981' : '#d1d5db'}
          strokeWidth="2"
          className="transition-all duration-300"
        />
      ))
    ).flat();
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700 p-8 transition-colors duration-200">
      <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-6 transition-colors duration-200">Network</h2>
      
      <div className="relative bg-gray-50 dark:bg-gray-900 rounded-xl p-6 h-80 mb-6 transition-colors duration-200">
        <svg className="absolute inset-6 w-full h-full" style={{ width: 'calc(100% - 48px)', height: 'calc(100% - 48px)' }}>
          {renderConnections()}
        </svg>
        
        {nodes.map((node) => (
          <div
            key={node.id}
            className={`absolute cursor-pointer transition-all duration-200 hover:scale-105 ${
              selectedNode === node.id ? 'z-10' : ''
            }`}
            style={{
              left: node.position.x - 32,
              top: node.position.y - 32
            }}
            onClick={() => onNodeSelect(node.id)}
          >
            <div
              className={`w-16 h-16 rounded-full flex items-center justify-center shadow-lg border-4 transition-all duration-200 ${
                !node.isOnline 
                  ? 'bg-gray-400 border-gray-300' 
                  : node.isMining 
                    ? 'bg-yellow-500 border-yellow-400 animate-pulse' 
                    : 'bg-emerald-500 border-emerald-400'
              } ${
                selectedNode === node.id ? 'border-blue-500 scale-110' : ''
              }`}
            >
              <Circle className="w-6 h-6 text-white" />
            </div>
            
            <div className="text-center mt-2">
              <div className="font-medium text-sm text-gray-700 dark:text-gray-300 transition-colors duration-200">{node.name}</div>
              <div className="text-xs text-gray-500 dark:text-gray-400 transition-colors duration-200">{node.blockchain.length} blocks</div>
            </div>
          </div>
        ))}
      </div>
      
      {selectedNode && (
        <div className="flex gap-3">
          <button
            onClick={() => onToggleNode(selectedNode)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
              nodes.find(n => n.id === selectedNode)?.isOnline
                ? 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 hover:bg-red-200 dark:hover:bg-red-900/50'
                : 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 hover:bg-green-200 dark:hover:bg-green-900/50'
            }`}
          >
            {nodes.find(n => n.id === selectedNode)?.isOnline ? (
              <>
                <PowerOff className="w-4 h-4" />
                Offline
              </>
            ) : (
              <>
                <Power className="w-4 h-4" />
                Online
              </>
            )}
          </button>
          
          <button
            onClick={() => onStartMining(selectedNode)}
            disabled={!nodes.find(n => n.id === selectedNode)?.isOnline}
            className="flex items-center gap-2 px-4 py-2 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 hover:bg-blue-200 dark:hover:bg-blue-900/50 disabled:bg-gray-100 dark:disabled:bg-gray-700 disabled:text-gray-400 dark:disabled:text-gray-500 rounded-lg font-medium transition-colors"
          >
            <Zap className="w-4 h-4" />
            Mine Block
          </button>
        </div>
      )}
    </div>
  );
};