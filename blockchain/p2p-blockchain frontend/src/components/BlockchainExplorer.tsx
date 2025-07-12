import React from 'react';
import { Node, Block } from '../types/blockchain';
import { Hash, Clock, User } from 'lucide-react';

interface BlockchainExplorerProps {
  selectedNode: Node | null;
}

export const BlockchainExplorer: React.FC<BlockchainExplorerProps> = ({ selectedNode }) => {
  if (!selectedNode) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700 p-8 transition-colors duration-200">
        <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 transition-colors duration-200">Blockchain</h2>
        <div className="text-center py-12">
          <div className="w-16 h-16 bg-gray-100 dark:bg-gray-700 rounded-full flex items-center justify-center mx-auto mb-4 transition-colors duration-200">
            <Hash className="w-8 h-8 text-gray-400 dark:text-gray-500 transition-colors duration-200" />
          </div>
          <p className="text-gray-500 dark:text-gray-400 transition-colors duration-200">Select a node to view its blockchain</p>
        </div>
      </div>
    );
  }

  const formatHash = (hash: string) => {
    return `${hash.substring(0, 6)}...${hash.substring(hash.length - 6)}`;
  };

  const formatTime = (timestamp: number) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700 p-8 transition-colors duration-200">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-semibold text-gray-900 dark:text-white transition-colors duration-200">Blockchain</h2>
        <div className="text-sm text-gray-500 dark:text-gray-400 transition-colors duration-200">
          {selectedNode.name} • {selectedNode.blockchain.length} blocks
        </div>
      </div>

      <div className="space-y-4 max-h-96 overflow-y-auto">
        {selectedNode.blockchain.slice().reverse().map((block: Block, index) => (
          <div
            key={`${block.index}-${block.hash}`}
            className={`border rounded-xl p-4 transition-all duration-200 ${
              block.index === 0 
                ? 'border-yellow-200 dark:border-yellow-800 bg-yellow-50 dark:bg-yellow-900/20' 
                : 'border-gray-200 dark:border-gray-600 hover:border-gray-300 dark:hover:border-gray-500'
            }`}
          >
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-3">
                <div className={`w-3 h-3 rounded-full ${
                  block.index === 0 ? 'bg-yellow-500' : 'bg-emerald-500'
                }`} />
                <span className="font-semibold text-gray-900 dark:text-white transition-colors duration-200">
                  Block #{block.index}
                </span>
              </div>
              
              <div className="flex items-center gap-4 text-sm text-gray-500 dark:text-gray-400 transition-colors duration-200">
                <div className="flex items-center gap-1">
                  <Clock className="w-4 h-4" />
                  {formatTime(block.timestamp)}
                </div>
                <span className="bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 px-2 py-1 rounded-full text-xs font-medium transition-colors duration-200">
                  {block.transactions.length} txs
                </span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-gray-500 dark:text-gray-400 transition-colors duration-200">Hash:</span>
                <div className="font-mono text-gray-900 dark:text-gray-100 transition-colors duration-200">{formatHash(block.hash)}</div>
              </div>
              <div>
                <span className="text-gray-500 dark:text-gray-400 transition-colors duration-200">Previous:</span>
                <div className="font-mono text-gray-900 dark:text-gray-100 transition-colors duration-200">{formatHash(block.previousHash)}</div>
              </div>
              <div>
                <span className="text-gray-500 dark:text-gray-400 transition-colors duration-200">Nonce:</span>
                <div className="font-mono text-gray-900 dark:text-gray-100 transition-colors duration-200">{block.nonce}</div>
              </div>
              <div>
                <span className="text-gray-500 dark:text-gray-400 transition-colors duration-200">Miner:</span>
                <div className="flex items-center gap-1 font-mono text-gray-900 dark:text-gray-100 transition-colors duration-200">
                  <User className="w-3 h-3" />
                  {block.miner}
                </div>
              </div>
            </div>

            {block.transactions.length > 0 && (
              <div className="mt-4 pt-4 border-t border-gray-100 dark:border-gray-600 transition-colors duration-200">
                <div className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 transition-colors duration-200">Transactions</div>
                <div className="space-y-2">
                  {block.transactions.map((tx, txIndex) => (
                    <div key={`${tx.id}-${txIndex}`} className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3 transition-colors duration-200">
                      <div className="flex justify-between items-center">
                        <div className="text-sm">
                          <span className="font-medium text-gray-900 dark:text-gray-100 transition-colors duration-200">{tx.from}</span>
                          <span className="text-gray-500 dark:text-gray-400 mx-2 transition-colors duration-200">→</span>
                          <span className="font-medium text-gray-900 dark:text-gray-100 transition-colors duration-200">{tx.to}</span>
                        </div>
                        <span className="font-semibold text-emerald-600 dark:text-emerald-400 transition-colors duration-200">
                          {tx.amount} coins
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};