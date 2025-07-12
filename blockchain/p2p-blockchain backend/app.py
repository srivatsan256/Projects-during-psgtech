from flask import Flask, jsonify, request, send_from_directory  # type: ignore
from flask_cors import CORS  # type: ignore
from blockchain import Blockchain, Transaction
from network import Network
import uuid
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

node_identifier = str(uuid.uuid4()).replace('-', '')
blockchain = Blockchain()
network = None  # Initialize as None, will assign later after parsing port

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
    
    transaction = Transaction(values['sender'], values['recipient'], values['amount'])
    blockchain.add_transaction(values['sender'], values['recipient'], values['amount'])
    return jsonify({
        'message': 'Transaction added',
        'transaction': transaction.to_dict()
    }), 201

@app.route('/api/mine', methods=['GET'])
def mine():
    if not blockchain.pending_transactions:
        return jsonify({'message': 'No pending transactions to mine'}), 400
    
    block = blockchain.mine_block()
    if network:
        network.consensus()
    
    response = {
        'message': 'New block mined',
        'index': block.index,
        'transactions': [tx.to_dict() for tx in block.transactions],
        'hash': block.hash,
        'previous_hash': block.previous_hash,
        'nonce': block.nonce
    }
    return jsonify(response), 200

@app.route('/api/chain', methods=['GET'])
def get_chain():
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
    
    return jsonify({
        'message': 'Nodes registered', 
        'total_nodes': list(blockchain.nodes)
    }), 201

@app.route('/api/nodes/sync', methods=['GET'])
def sync():
    if not network:
        return jsonify({'message': 'Network not initialized'}), 400
        
    replaced = network.consensus()
    if replaced:
        return jsonify({
            'message': 'Chain replaced', 
            'new_chain': [vars(block) for block in blockchain.chain]
        }), 200
    
    return jsonify({
        'message': 'Chain is up to date', 
        'chain': [vars(block) for block in blockchain.chain]
    }), 200

@app.route('/api/stats', methods=['GET'])
def get_stats():
    stats = {
        'total_blocks': len(blockchain.chain),
        'pending_transactions': len(blockchain.pending_transactions),
        'difficulty': blockchain.difficulty,
        'total_nodes': len(blockchain.nodes),
        'node_identifier': node_identifier
    }
    return jsonify(stats), 200

@app.route('/api/validate', methods=['GET'])
def validate_chain():
    is_valid = blockchain.is_chain_valid()
    return jsonify({
        'valid': is_valid,
        'message': 'Chain is valid' if is_valid else 'Chain is invalid'
    }), 200

# Legacy route for backward compatibility
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

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=5000, type=int, help='port to listen on')
    args = parser.parse_args()

    # Now create the network with the correct port
    network = Network(blockchain, f'localhost:{args.port}')

    app.run(host='0.0.0.0', port=args.port, debug=True)
