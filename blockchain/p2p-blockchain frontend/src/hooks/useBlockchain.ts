import { useState, useEffect, useCallback } from 'react';
import { NetworkSimulation } from '../network/NetworkSimulation';
import { Node, NetworkStats } from '../types/blockchain';

export const useBlockchain = () => {
  const [network] = useState(() => new NetworkSimulation());
  const [nodes, setNodes] = useState<Node[]>([]);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [networkStats, setNetworkStats] = useState<NetworkStats>({
    totalNodes: 0,
    onlineNodes: 0,
    totalTransactions: 0,
    networkHashRate: 0,
    difficulty: 1
  });

  const updateNodes = useCallback(() => {
    setNodes(network.getNodes());
    setNetworkStats(network.getNetworkStats());
  }, [network]);

  useEffect(() => {
    updateNodes();
    const interval = setInterval(updateNodes, 1000);
    return () => clearInterval(interval);
  }, [updateNodes]);

  const createTransaction = useCallback(async (from: string, to: string, amount: number, fromNodeId: string) => {
    const transaction = network.createTransaction(from, to, amount);
    await network.broadcastTransaction(transaction, fromNodeId);
    updateNodes();
  }, [network, updateNodes]);

  const startMining = useCallback(async (nodeId: string) => {
    await network.startMining(nodeId);
    updateNodes();
  }, [network, updateNodes]);

  const toggleNodeStatus = useCallback((nodeId: string) => {
    network.toggleNodeStatus(nodeId);
    updateNodes();
  }, [network, updateNodes]);

  const syncAllNodes = useCallback(async () => {
    await network.syncAllNodes();
    updateNodes();
  }, [network, updateNodes]);

  const getSelectedNodeData = useCallback(() => {
    if (!selectedNode) return null;
    return network.getNode(selectedNode);
  }, [network, selectedNode]);

  return {
    nodes,
    selectedNode,
    setSelectedNode,
    networkStats,
    createTransaction,
    startMining,
    toggleNodeStatus,
    syncAllNodes,
    getSelectedNodeData
  };
};