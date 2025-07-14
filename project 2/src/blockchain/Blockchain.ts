import { Block, Transaction } from '../types/blockchain';

export class Blockchain {
  private chain: Block[] = [];
  private difficulty: number = 2;

  constructor() {
    this.chain = [this.createGenesisBlock()];
  }

  private createGenesisBlock(): Block {
    return {
      index: 0,
      timestamp: Date.now(),
      transactions: [],
      previousHash: '0',
      hash: this.calculateHash(0, Date.now(), [], '0', 0),
      nonce: 0,
      miner: 'genesis',
      difficulty: this.difficulty
    };
  }

  private calculateHash(index: number, timestamp: number, transactions: Transaction[], previousHash: string, nonce: number): string {
    const data = `${index}${timestamp}${JSON.stringify(transactions)}${previousHash}${nonce}`;
    return this.sha256(data);
  }

  private sha256(data: string): string {
    let hash = 0;
    for (let i = 0; i < data.length; i++) {
      const char = data.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return Math.abs(hash).toString(16).padStart(8, '0');
  }

  getLatestBlock(): Block {
    return this.chain[this.chain.length - 1];
  }

  async mineBlock(transactions: Transaction[], minerAddress: string): Promise<Block> {
    const previousBlock = this.getLatestBlock();
    const newBlock: Block = {
      index: previousBlock.index + 1,
      timestamp: Date.now(),
      transactions: [...transactions],
      previousHash: previousBlock.hash,
      hash: '',
      nonce: 0,
      miner: minerAddress,
      difficulty: this.difficulty
    };

    const target = '0'.repeat(this.difficulty);

    while (true) {
      newBlock.hash = this.calculateHash(
        newBlock.index,
        newBlock.timestamp,
        newBlock.transactions,
        newBlock.previousHash,
        newBlock.nonce
      );

      if (newBlock.hash.startsWith(target)) {
        break;
      }

      newBlock.nonce++;

      if (newBlock.nonce % 1000 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }

    return newBlock;
  }

  addBlock(block: Block): boolean {
    if (this.isValidBlock(block)) {
      this.chain.push(block);
      return true;
    }
    return false;
  }

  private isValidBlock(block: Block): boolean {
    const previousBlock = this.getLatestBlock();
    
    if (block.index !== previousBlock.index + 1) return false;
    if (block.previousHash !== previousBlock.hash) return false;
    
    const calculatedHash = this.calculateHash(
      block.index,
      block.timestamp,
      block.transactions,
      block.previousHash,
      block.nonce
    );
    
    return block.hash === calculatedHash && block.hash.startsWith('0'.repeat(this.difficulty));
  }

  getChain(): Block[] {
    return [...this.chain];
  }

  replaceChain(newChain: Block[]): boolean {
    if (newChain.length > this.chain.length) {
      this.chain = [...newChain];
      return true;
    }
    return false;
  }

  getDifficulty(): number {
    return this.difficulty;
  }
}