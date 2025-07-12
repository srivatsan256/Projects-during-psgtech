import { useState, useEffect, useCallback } from 'react';
import { blockchainApi, ApiBlock, ApiTransaction } from '../services/api';
import { Node, NetworkStats, Block, Transaction } from '../types/blockchain';

export const useBlockchainApi = () => {
  const [chain, setChain] = useState<Block[]>([]);
  const [pendingTransactions, setPendingTransactions] = useState<Transaction[]>([]);
  const [stats, setStats] = useState<NetworkStats>({
    totalNodes: 1,
    onlineNodes: 1,
    totalTransactions: 0,
    networkHashRate: 0,
    difficulty: 1
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Convert API data to frontend types
  const convertApiBlockToBlock = (apiBlock: ApiBlock): Block => ({
    index: apiBlock.index,
    transactions: apiBlock.transactions.map(tx => ({
      id: `tx-${tx.timestamp}`,
      from: tx.sender,
      to: tx.recipient,
      amount: tx.amount,
      timestamp: tx.timestamp
    })),
    timestamp: apiBlock.timestamp,
    hash: apiBlock.hash,
    previousHash: apiBlock.previous_hash,
    nonce: apiBlock.nonce,
    miner: 'main-node',
    difficulty: 4
  });

  const convertApiTransactionToTransaction = (apiTx: ApiTransaction): Transaction => ({
    id: `tx-${apiTx.timestamp}`,
    from: apiTx.sender,
    to: apiTx.recipient,
    amount: apiTx.amount,
    timestamp: apiTx.timestamp
  });

  const fetchChain = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      const response = await blockchainApi.getChain();
      const convertedChain = response.chain.map(convertApiBlockToBlock);
      setChain(convertedChain);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch chain');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const fetchPendingTransactions = useCallback(async () => {
    try {
      const response = await blockchainApi.getPendingTransactions();
      const convertedTransactions = response.pending_transactions.map(convertApiTransactionToTransaction);
      setPendingTransactions(convertedTransactions);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch pending transactions');
    }
  }, []);

  const fetchStats = useCallback(async () => {
    try {
      const response = await blockchainApi.getStats();
      setStats({
        totalNodes: response.total_nodes,
        onlineNodes: response.total_nodes, // Assuming all nodes are online for now
        totalTransactions: chain.reduce((sum: number, block: Block) => sum + block.transactions.length, 0),
        networkHashRate: 100, // Mock value
        difficulty: response.difficulty
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch stats');
    }
  }, [chain]);

  const createTransaction = useCallback(async (from: string, to: string, amount: number) => {
    try {
      setIsLoading(true);
      setError(null);
      await blockchainApi.createTransaction(from, to, amount);
      await fetchPendingTransactions();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create transaction');
    } finally {
      setIsLoading(false);
    }
  }, [fetchPendingTransactions]);

  const mineBlock = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      await blockchainApi.mineBlock();
      await fetchChain();
      await fetchPendingTransactions();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to mine block');
    } finally {
      setIsLoading(false);
    }
  }, [fetchChain, fetchPendingTransactions]);

  const validateChain = useCallback(async () => {
    try {
      const response = await blockchainApi.validateChain();
      return response.valid;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to validate chain');
      return false;
    }
  }, []);

  const registerNodes = useCallback(async (nodes: string[]) => {
    try {
      setIsLoading(true);
      setError(null);
      await blockchainApi.registerNodes(nodes);
      await fetchStats();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to register nodes');
    } finally {
      setIsLoading(false);
    }
  }, [fetchStats]);

  const syncNodes = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      await blockchainApi.syncNodes();
      await fetchChain();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to sync nodes');
    } finally {
      setIsLoading(false);
    }
  }, [fetchChain]);

  useEffect(() => {
    fetchChain();
    fetchPendingTransactions();
    fetchStats();
  }, [fetchChain, fetchPendingTransactions, fetchStats]);

  // Create a mock node for compatibility with the existing UI
  const mockNode: Node = {
    id: 'main-node',
    name: 'Main Node',
    blockchain: chain,
    pendingTransactions,
    isOnline: true,
    isMining: isLoading,
    position: { x: 300, y: 150 }
  };

  return {
    // Original interface for compatibility
    nodes: [mockNode],
    selectedNode: 'main-node',
    setSelectedNode: () => {},
    networkStats: stats,
    createTransaction: (from: string, to: string, amount: number, nodeId: string) => createTransaction(from, to, amount),
    startMining: mineBlock,
    toggleNodeStatus: () => {},
    syncAllNodes: syncNodes,
    getSelectedNodeData: () => mockNode,
    
    // New API-specific methods
    chain,
    pendingTransactions,
    isLoading,
    error,
    validateChain,
    registerNodes,
    refreshData: () => {
      fetchChain();
      fetchPendingTransactions();
      fetchStats();
    }
  };
};