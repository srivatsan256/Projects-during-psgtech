import { Transaction, Block } from '../types/blockchain';

const API_BASE_URL = 'http://localhost:5000/api';

export interface ApiTransaction {
  sender: string;
  recipient: string;
  amount: number;
  timestamp: number;
}

export interface ApiBlock {
  index: number;
  transactions: ApiTransaction[];
  timestamp: number;
  previous_hash: string;
  hash: string;
  nonce: number;
}

export interface ApiChainResponse {
  chain: ApiBlock[];
  length: number;
}

export interface ApiStatsResponse {
  total_blocks: number;
  pending_transactions: number;
  difficulty: number;
  total_nodes: number;
  node_identifier: string;
}

export interface ApiPendingResponse {
  pending_transactions: ApiTransaction[];
  count: number;
}

export interface ApiMineResponse {
  message: string;
  index: number;
  transactions: ApiTransaction[];
  hash: string;
  previous_hash: string;
  nonce: number;
}

export class BlockchainApi {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  async getChain(): Promise<ApiChainResponse> {
    const response = await fetch(`${this.baseUrl}/chain`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  }

  async getPendingTransactions(): Promise<ApiPendingResponse> {
    const response = await fetch(`${this.baseUrl}/pending`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  }

  async createTransaction(sender: string, recipient: string, amount: number): Promise<{ message: string; transaction: ApiTransaction }> {
    const response = await fetch(`${this.baseUrl}/transaction/new`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        sender,
        recipient,
        amount,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  }

  async mineBlock(): Promise<ApiMineResponse> {
    const response = await fetch(`${this.baseUrl}/mine`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  }

  async getStats(): Promise<ApiStatsResponse> {
    const response = await fetch(`${this.baseUrl}/stats`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  }

  async validateChain(): Promise<{ valid: boolean; message: string }> {
    const response = await fetch(`${this.baseUrl}/validate`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  }

  async registerNodes(nodes: string[]): Promise<{ message: string; total_nodes: string[] }> {
    const response = await fetch(`${this.baseUrl}/nodes/register`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ nodes }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  }

  async syncNodes(): Promise<{ message: string; new_chain?: ApiBlock[]; chain?: ApiBlock[] }> {
    const response = await fetch(`${this.baseUrl}/nodes/sync`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  }
}

export const blockchainApi = new BlockchainApi();