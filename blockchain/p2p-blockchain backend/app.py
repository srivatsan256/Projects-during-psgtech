from flask import Flask, jsonify, request, send_from_directory  # type: ignore
from flask_cors import CORS  # type: ignore
from blockchain import Blockchain, Transaction
from network import Network
import uuid
import os
import threading
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

node_identifier = str(uuid.uuid4()).replace('-', '')
blockchain = Blockchain()
network = None  # Initialize as None, will assign later after parsing port
blockchain_lock = threading.Lock()

# Serve React frontend
@app.route('/')
def serve_react():
    return send_from_directory('../p2p-blockchain frontend/dist', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('../p2p-blockchain frontend/dist', path)

@app.route('/api/transaction/new', methods=['POST'])
def new_transaction():
    values = request.get_json()
    required = ['sender', 'recipient', 'amount']
    if not all(k in values for k in required):
        return jsonify({'message': 'Missing values'}), 400
    
    with blockchain_lock:
        transaction = Transaction(values['sender'], values['recipient'], values['amount'])
        blockchain.add_transaction(values['sender'], values['recipient'], values['amount'])
        
        # Broadcast transaction to all peers
        if network:
            try:
                network.broadcast_transaction(transaction)
            except Exception as e:
                print(f"Failed to broadcast transaction: {e}")
    
    return jsonify({
        'message': 'Transaction added and broadcasted',
        'transaction': transaction.to_dict()
    }), 201

@app.route('/api/mine', methods=['GET'])
def mine():
    with blockchain_lock:
        if not blockchain.pending_transactions:
            return jsonify({'message': 'No pending transactions to mine'}), 400
        
        print(f"Mining block with {len(blockchain.pending_transactions)} transactions...")
        block = blockchain.mine_block()
        
        # Broadcast block to all peers
        if network:
            try:
                network.broadcast_block(block)
            except Exception as e:
                print(f"Failed to broadcast block: {e}")
        
        print(f"Block #{block.index} mined successfully with hash: {block.hash}")
    
    # Trigger consensus after mining
    if network:
        try:
            network.consensus()
        except Exception as e:
            print(f"Consensus failed: {e}")
    
    response = {
        'message': 'New block mined and broadcasted',
        'index': block.index,
        'transactions': [tx.to_dict() for tx in block.transactions],
        'hash': block.hash,
        'previous_hash': block.previous_hash,
        'nonce': block.nonce
    }
    return jsonify(response), 200

@app.route('/api/chain', methods=['GET'])
def get_chain():
    with blockchain_lock:
        chain_data = []
        for block in blockchain.chain:
            block_data = {
                'index': block.index,
                'transactions': [tx.to_dict() for tx in block.transactions],
                'timestamp': block.timestamp,
                'previous_hash': block.previous_hash,
                'hash': block.hash,
                'nonce': block.nonce
            }
            chain_data.append(block_data)
        
        response = {
            'chain': chain_data,
            'length': len(blockchain.chain)
        }
    return jsonify(response), 200

@app.route('/api/pending', methods=['GET'])
def get_pending_transactions():
    with blockchain_lock:
        response = {
            'pending_transactions': [tx.to_dict() for tx in blockchain.pending_transactions],
            'count': len(blockchain.pending_transactions)
        }
    return jsonify(response), 200

@app.route('/api/nodes/register', methods=['POST'])
def register_nodes():
    values = request.get_json()
    nodes = values.get('nodes')
    if nodes is None:
        return jsonify({'message': 'Invalid node list'}), 400
    
    if network:
        for node in nodes:
            network.register_node(node)
        
        # Start periodic sync if not already started
        if not hasattr(network, '_sync_started'):
            network.start_periodic_sync()
            network._sync_started = True
    
    return jsonify({
        'message': 'Nodes registered and sync started', 
        'total_nodes': len(blockchain.nodes),
        'nodes': list(blockchain.nodes)
    }), 201

@app.route('/api/nodes/sync', methods=['GET', 'POST'])
def sync():
    if not network:
        return jsonify({'message': 'Network not initialized'}), 400
    
    try:
        replaced = network.consensus()
        if replaced:
            return jsonify({
                'message': 'Chain replaced with longer valid chain', 
                'new_length': len(blockchain.chain)
            }), 200
        
        return jsonify({
            'message': 'Chain is already up to date', 
            'length': len(blockchain.chain)
        }), 200
    except Exception as e:
        return jsonify({
            'message': f'Sync failed: {str(e)}',
            'error': True
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    with blockchain_lock:
        stats = {
            'total_blocks': len(blockchain.chain),
            'pending_transactions': len(blockchain.pending_transactions),
            'difficulty': blockchain.difficulty,
            'total_nodes': len(blockchain.nodes),
            'node_identifier': node_identifier,
            'node_address': network.node_address if network else 'unknown',
            'connected_peers': list(blockchain.nodes) if blockchain.nodes else []
        }
    return jsonify(stats), 200

@app.route('/api/validate', methods=['GET'])
def validate_chain():
    with blockchain_lock:
        is_valid = blockchain.is_chain_valid()
    return jsonify({
        'valid': is_valid,
        'message': 'Chain is valid' if is_valid else 'Chain is invalid'
    }), 200

@app.route('/api/network/status', methods=['GET'])
def network_status():
    if not network:
        return jsonify({'message': 'Network not initialized'}), 400
    
    status = {
        'node_address': network.node_address,
        'total_peers': len(blockchain.nodes),
        'peers': list(blockchain.nodes),
        'last_sync': getattr(network, 'last_sync', 0),
        'sync_interval': network.sync_interval
    }
    return jsonify(status), 200

# Legacy routes for backward compatibility
@app.route('/transaction/new', methods=['POST'])
def new_transaction_legacy():
    return new_transaction()

@app.route('/mine', methods=['GET'])
def mine_legacy():
    return mine()

@app.route('/chain', methods=['GET'])
def get_chain_legacy():
    return get_chain()

@app.route('/nodes/register', methods=['POST'])
def register_nodes_legacy():
    return register_nodes()

@app.route('/nodes/sync', methods=['GET'])
def sync_legacy():
    return sync()

def initialize_network_monitoring():
    """Initialize network monitoring and periodic tasks"""
    def monitor_network():
        while True:
            try:
                if network and blockchain.nodes:
                    print(f"[{network.node_address}] Network status: {len(blockchain.nodes)} peers, {len(blockchain.chain)} blocks")
                time.sleep(30)  # Monitor every 30 seconds
            except Exception as e:
                print(f"Network monitoring error: {e}")
                time.sleep(30)
    
    monitor_thread = threading.Thread(target=monitor_network, daemon=True)
    monitor_thread.start()

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=5000, type=int, help='port to listen on')
    args = parser.parse_args()

    # Create the network with the correct port
    network = Network(blockchain, f'localhost:{args.port}')
    
    # Initialize network monitoring
    initialize_network_monitoring()
    
    print(f"Starting blockchain node on port {args.port}")
    print(f"Node identifier: {node_identifier}")
    print(f"Node address: {network.node_address}")

    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)
