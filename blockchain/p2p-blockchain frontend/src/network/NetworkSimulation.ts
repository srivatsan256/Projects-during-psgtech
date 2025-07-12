import { Node, Transaction, Block, NetworkStats } from '../types/blockchain';
import { Blockchain } from '../blockchain/Blockchain';

export class NetworkSimulation {
  private nodes: Map<string, Node> = new Map();
  private blockchain: Blockchain = new Blockchain();

  constructor() {
    this.initializeNetwork();
  }

  private initializeNetwork(): void {
    const nodeConfigs = [
      { name: 'Node A', position: { x: 200, y: 80 } },
      { name: 'Node B', position: { x: 350, y: 80 } },
      { name: 'Node C', position: { x: 500, y: 80 } },
      { name: 'Node D', position: { x: 275, y: 200 } },
      { name: 'Node E', position: { x: 425, y: 200 } }
    ];

    nodeConfigs.forEach(config => {
      this.addNode(config.name, config.position);
    });
  }

  addNode(name: string, position: { x: number; y: number }): string {
    const nodeId = `node-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const node: Node = {
      id: nodeId,
      name,
      blockchain: this.blockchain.getChain(),
      pendingTransactions: [],
      isOnline: true,
      isMining: false,
      position
    };

    this.nodes.set(nodeId, node);
    return nodeId;
  }

  async broadcastTransaction(transaction: Transaction, fromNodeId: string): Promise<void> {
    const fromNode = this.nodes.get(fromNodeId);
    if (!fromNode || !fromNode.isOnline) return;

    // Add to all online nodes
    this.nodes.forEach(node => {
      if (node.isOnline) {
        node.pendingTransactions.push(transaction);
      }
    });
  }

  async startMining(nodeId: string): Promise<void> {
    const node = this.nodes.get(nodeId);
    if (!node || !node.isOnline || node.isMining) return;

    node.isMining = true;

    try {
      const transactions = node.pendingTransactions.splice(0, 3);
      if (transactions.length === 0) {
        node.isMining = false;
        return;
      }

      const newBlock = await this.blockchain.mineBlock(transactions, nodeId);
      
      // Add block to all nodes
      this.nodes.forEach(n => {
        if (n.isOnline) {
          n.blockchain.push(newBlock);
          // Remove mined transactions
          n.pendingTransactions = n.pendingTransactions.filter(
            tx => !transactions.some(minedTx => minedTx.id === tx.id)
          );
        }
      });

    } catch (error) {
      console.error('Mining error:', error);
    } finally {
      node.isMining = false;
    }
  }

  getNodes(): Node[] {
    return Array.from(this.nodes.values());
  }

  getNode(nodeId: string): Node | undefined {
    return this.nodes.get(nodeId);
  }

  toggleNodeStatus(nodeId: string): void {
    const node = this.nodes.get(nodeId);
    if (node) {
      node.isOnline = !node.isOnline;
      if (!node.isOnline) {
        node.isMining = false;
      }
    }
  }

  getNetworkStats(): NetworkStats {
    const nodes = Array.from(this.nodes.values());
    const onlineNodes = nodes.filter(n => n.isOnline);
    
    const totalTransactions = nodes.reduce((sum, node) => {
      return sum + node.blockchain.reduce((txSum, block) => txSum + block.transactions.length, 0);
    }, 0);

    const networkHashRate = onlineNodes.filter(n => n.isMining).length * 500;

    return {
      totalNodes: nodes.length,
      onlineNodes: onlineNodes.length,
      totalTransactions,
      networkHashRate,
      difficulty: this.blockchain.getDifficulty()
    };
  }

  createTransaction(from: string, to: string, amount: number): Transaction {
    return {
      id: `tx-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      from,
      to,
      amount,
      timestamp: Date.now()
    };
  }

  async syncAllNodes(): Promise<void> {
    // Find longest chain
    let longestChain = this.blockchain.getChain();
    this.nodes.forEach(node => {
      if (node.blockchain.length > longestChain.length) {
        longestChain = node.blockchain;
      }
    });

    // Update all nodes with longest chain
    this.nodes.forEach(node => {
      if (node.isOnline) {
        node.blockchain = [...longestChain];
      }
    });
  }
}