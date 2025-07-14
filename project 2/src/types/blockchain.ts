export interface Transaction {
  id: string;
  from: string;
  to: string;
  amount: number;
  timestamp: number;
}

export interface Block {
  index: number;
  timestamp: number;
  transactions: Transaction[];
  previousHash: string;
  hash: string;
  nonce: number;
  miner: string;
  difficulty: number;
}

export interface Node {
  id: string;
  name: string;
  blockchain: Block[];
  pendingTransactions: Transaction[];
  isOnline: boolean;
  isMining: boolean;
  position: { x: number; y: number };
}

export interface NetworkStats {
  totalNodes: number;
  onlineNodes: number;
  totalTransactions: number;
  networkHashRate: number;
  difficulty: number;
}