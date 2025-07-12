import requests  # type: ignore
import json
import sys
import argparse
from blockchain import Block, Blockchain


class Network:
    def __init__(self, blockchain, node_address):
        self.blockchain = blockchain
        self.node_address = node_address

    def register_node(self, address):
        self.blockchain.nodes.add(address)

    def get_chain_from_peer(self, address):
        try:
            response = requests.get(f'http://{address}/chain')
            if response.status_code == 200:
                return response.json()
        except requests.RequestException:
            return None

    def consensus(self):
        longest_chain = None
        max_length = len(self.blockchain.chain)

        for node in self.blockchain.nodes:
            chain_data = self.get_chain_from_peer(node)
            if chain_data and len(chain_data['chain']) > max_length:
                chain = [Block(**block) for block in chain_data['chain']]
                temp_blockchain = Blockchain()
                temp_blockchain.chain = chain
                if temp_blockchain.is_chain_valid():
                    longest_chain = chain
                    max_length = len(chain)

        if longest_chain:
            self.blockchain.chain = longest_chain
            return True
        return False


# --- Argument parsing ---
parser = argparse.ArgumentParser(description="P2P Blockchain Node")
parser.add_argument('--port', type=int, default=5000, help='Port number to run the server on')
args = parser.parse_args()

PORT = args.port

# --- Initialize blockchain and network ---
blockchain = Blockchain()
network = Network(blockchain, f'localhost:{PORT}')
