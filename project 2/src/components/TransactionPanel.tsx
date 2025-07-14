import React, { useState } from 'react';
import { Send, Plus, Clock } from 'lucide-react';

interface TransactionPanelProps {
  nodes: any[];
  selectedNode: string | null;
  onCreateTransaction: (from: string, to: string, amount: number, fromNodeId: string) => void;
}

export const TransactionPanel: React.FC<TransactionPanelProps> = ({
  nodes,
  selectedNode,
  onCreateTransaction
}) => {
  const [fromAddress, setFromAddress] = useState('');
  const [toAddress, setToAddress] = useState('');
  const [amount, setAmount] = useState('');
  const [isCreating, setIsCreating] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedNode || !fromAddress || !toAddress || !amount) return;

    setIsCreating(true);
    try {
      await onCreateTransaction(fromAddress, toAddress, parseFloat(amount), selectedNode);
      setFromAddress('');
      setToAddress('');
      setAmount('');
    } finally {
      setIsCreating(false);
    }
  };

  const selectedNodeData = nodes.find(n => n.id === selectedNode);
  const pendingTransactions = selectedNodeData?.pendingTransactions || [];

  return (
    <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700 p-8 transition-colors duration-200">
      <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-6 transition-colors duration-200">Transactions</h2>
      
      <div className="space-y-8">
        {/* Create Transaction */}
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6 transition-colors duration-200">
          <h3 className="font-semibold text-blue-900 dark:text-blue-100 mb-4 flex items-center gap-2 transition-colors duration-200">
            <Plus className="w-5 h-5" />
            New Transaction
          </h3>
          
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 transition-colors duration-200">
                  From
                </label>
                <input
                  type="text"
                  value={fromAddress}
                  onChange={(e) => setFromAddress(e.target.value)}
                  placeholder="Sender address"
                  className="w-full px-4 py-3 border border-gray-200 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                  required
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 transition-colors duration-200">
                  To
                </label>
                <input
                  type="text"
                  value={toAddress}
                  onChange={(e) => setToAddress(e.target.value)}
                  placeholder="Recipient address"
                  className="w-full px-4 py-3 border border-gray-200 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                  required
                />
              </div>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 transition-colors duration-200">
                Amount
              </label>
              <input
                type="number"
                value={amount}
                onChange={(e) => setAmount(e.target.value)}
                placeholder="0.00"
                min="0"
                step="0.01"
                className="w-full px-4 py-3 border border-gray-200 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                required
              />
            </div>
            
            <button
              type="submit"
              disabled={!selectedNode || isCreating}
              className="w-full bg-blue-600 hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 disabled:bg-gray-300 dark:disabled:bg-gray-600 text-white font-semibold py-3 px-4 rounded-lg transition-colors duration-200 flex items-center justify-center gap-2"
            >
              {isCreating ? (
                <div className="animate-spin w-5 h-5 border-2 border-white border-t-transparent rounded-full" />
              ) : (
                <Send className="w-5 h-5" />
              )}
              {isCreating ? 'Creating...' : 'Send Transaction'}
            </button>
          </form>
          
          {!selectedNode && (
            <p className="mt-4 text-sm text-gray-500 dark:text-gray-400 text-center transition-colors duration-200">
              Select a node to create transactions
            </p>
          )}
        </div>

        {/* Pending Transactions */}
        <div>
          <h3 className="font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2 transition-colors duration-200">
            <Clock className="w-5 h-5" />
            Pending
            {pendingTransactions.length > 0 && (
              <span className="bg-yellow-100 text-yellow-800 px-2 py-1 rounded-full text-sm">
                {pendingTransactions.length}
              </span>
            )}
          </h3>
          
          <div className="space-y-3 max-h-64 overflow-y-auto">
            {pendingTransactions.length === 0 ? (
              <div className="text-center py-8">
                <div className="w-12 h-12 bg-gray-100 dark:bg-gray-700 rounded-full flex items-center justify-center mx-auto mb-3 transition-colors duration-200">
                  <Clock className="w-6 h-6 text-gray-400 dark:text-gray-500 transition-colors duration-200" />
                </div>
                <p className="text-gray-500 dark:text-gray-400 transition-colors duration-200">No pending transactions</p>
              </div>
            ) : (
              pendingTransactions.map((tx: any, index: number) => (
                <div key={`${tx.id}-${index}`} className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4 transition-colors duration-200">
                  <div className="flex justify-between items-center">
                    <div>
                      <div className="text-sm">
                        <span className="font-medium text-gray-900 dark:text-gray-100 transition-colors duration-200">{tx.from}</span>
                        <span className="text-gray-500 dark:text-gray-400 mx-2 transition-colors duration-200">→</span>
                        <span className="font-medium text-gray-900 dark:text-gray-100 transition-colors duration-200">{tx.to}</span>
                      </div>
                      <div className="text-xs text-gray-500 dark:text-gray-400 mt-1 transition-colors duration-200">
                        {new Date(tx.timestamp).toLocaleTimeString()}
                      </div>
                    </div>
                    <div className="font-semibold text-yellow-700">
                      <div className="font-semibold text-yellow-700 dark:text-yellow-400 transition-colors duration-200">
                        {tx.amount} coins
                      </div>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
};