import requests  # type: ignore
import json
import sys
import argparse
import threading
import time
from blockchain import Block, Blockchain, Transaction


class Network:
    def __init__(self, blockchain, node_address):
        self.blockchain = blockchain
        self.node_address = node_address
        self.sync_lock = threading.Lock()
        self.last_sync = 0
        self.sync_interval = 5  # seconds

    def register_node(self, address):
        """Register a new node in the network"""
        if address != self.node_address:
            self.blockchain.nodes.add(address)
            print(f"Registered node: {address}")

    def get_chain_from_peer(self, address):
        """Get blockchain from a peer node"""
        try:
            response = requests.get(f'http://{address}/api/chain', timeout=5)
            if response.status_code == 200:
                return response.json()
        except requests.RequestException as e:
            print(f"Failed to get chain from {address}: {e}")
        return None

    def broadcast_transaction(self, transaction):
        """Broadcast a transaction to all peer nodes"""
        transaction_data = transaction.to_dict()
        
        def send_to_peer(address):
            try:
                requests.post(f'http://{address}/api/transaction/new', 
                            json=transaction_data, timeout=3)
            except requests.RequestException:
                pass  # Ignore failed broadcasts
        
        threads = []
        for node in self.blockchain.nodes:
            thread = threading.Thread(target=send_to_peer, args=(node,))
            threads.append(thread)
            thread.start()
        
        # Wait for all broadcasts to complete
        for thread in threads:
            thread.join(timeout=5)

    def broadcast_block(self, block):
        """Broadcast a mined block to all peer nodes"""
        def notify_peer(address):
            try:
                requests.post(f'http://{address}/api/nodes/sync', timeout=3)
            except requests.RequestException:
                pass  # Ignore failed notifications
        
        threads = []
        for node in self.blockchain.nodes:
            thread = threading.Thread(target=notify_peer, args=(node,))
            threads.append(thread)
            thread.start()
        
        # Wait for all notifications to complete
        for thread in threads:
            thread.join(timeout=5)

    def consensus(self):
        """Consensus algorithm - adopt the longest valid chain"""
        with self.sync_lock:
            current_time = time.time()
            if current_time - self.last_sync < self.sync_interval:
                return False  # Don't sync too frequently
            
            self.last_sync = current_time
            
            longest_chain = None
            max_length = len(self.blockchain.chain)
            
            # Get chains from all peers in parallel
            def get_peer_chain(address, results, index):
                chain_data = self.get_chain_from_peer(address)
                results[index] = (address, chain_data)
            
            peers = list(self.blockchain.nodes)
            results = [None] * len(peers)
            threads = []
            
            for i, node in enumerate(peers):
                thread = threading.Thread(target=get_peer_chain, args=(node, results, i))
                threads.append(thread)
                thread.start()
            
            # Wait for all requests to complete
            for thread in threads:
                thread.join(timeout=10)
            
            # Process results
            for result in results:
                if result and result[1]:
                    address, chain_data = result
                    if chain_data and len(chain_data['chain']) > max_length:
                        try:
                            # Reconstruct chain from JSON data
                            chain = []
                            for block_data in chain_data['chain']:
                                transactions = []
                                for tx_data in block_data['transactions']:
                                    tx = Transaction(tx_data['sender'], tx_data['recipient'], tx_data['amount'])
                                    tx.timestamp = tx_data['timestamp']
                                    transactions.append(tx)
                                
                                block = Block(block_data['index'], transactions, 
                                            block_data['timestamp'], block_data['previous_hash'])
                                block.nonce = block_data['nonce']
                                block.hash = block_data['hash']
                                chain.append(block)
                            
                            # Validate the chain
                            temp_blockchain = Blockchain()
                            temp_blockchain.chain = chain
                            if temp_blockchain.is_chain_valid():
                                longest_chain = chain
                                max_length = len(chain)
                                print(f"Found longer valid chain from {address}: {max_length} blocks")
                        except Exception as e:
                            print(f"Error processing chain from {address}: {e}")
            
            if longest_chain:
                self.blockchain.chain = longest_chain
                self.blockchain.pending_transactions = []  # Clear pending transactions
                print("Blockchain updated with longer chain")
                return True
            
            return False

    def start_periodic_sync(self):
        """Start periodic synchronization with peers"""
        def sync_worker():
            while True:
                time.sleep(self.sync_interval)
                if self.blockchain.nodes:
                    self.consensus()
        
        sync_thread = threading.Thread(target=sync_worker, daemon=True)
        sync_thread.start()


# Remove the global initialization to prevent conflicts
if __name__ == "__main__":
    # --- Argument parsing ---
    parser = argparse.ArgumentParser(description="P2P Blockchain Node")
    parser.add_argument('--port', type=int, default=5000, help='Port number to run the server on')
    args = parser.parse_args()

    PORT = args.port

    # --- Initialize blockchain and network ---
    blockchain = Blockchain()
    network = Network(blockchain, f'localhost:{PORT}')
    
    print(f"Network initialized on port {PORT}")
