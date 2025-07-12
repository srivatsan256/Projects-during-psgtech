from flask import Flask, jsonify, request  # type: ignore
from blockchain import Blockchain, Transaction
from network import Network
import uuid

app = Flask(__name__)
node_identifier = str(uuid.uuid4()).replace('-', '')
blockchain = Blockchain()
network = None  # Initialize as None, will assign later after parsing port


@app.route('/transaction/new', methods=['POST'])
def new_transaction():
    values = request.get_json()
    required = ['sender', 'recipient', 'amount']
    if not all(k in values for k in required):
        return jsonify({'message': 'Missing values'}), 400
    blockchain.add_transaction(values['sender'], values['recipient'], values['amount'])
    return jsonify({'message': 'Transaction added'}), 201


@app.route('/mine', methods=['GET'])
def mine():
    block = blockchain.mine_block()
    network.consensus()
    response = {
        'message': 'New block mined',
        'index': block.index,
        'transactions': [tx.to_dict() for tx in block.transactions],
        'hash': block.hash,
        'previous_hash': block.previous_hash
    }
    return jsonify(response), 200


@app.route('/chain', methods=['GET'])
def get_chain():
    response = {
        'chain': [vars(block) for block in blockchain.chain],
        'length': len(blockchain.chain)
    }
    return jsonify(response), 200


@app.route('/nodes/register', methods=['POST'])
def register_nodes():
    values = request.get_json()
    nodes = values.get('nodes')
    if nodes is None:
        return jsonify({'message': 'Invalid node list'}), 400
    for node in nodes:
        network.register_node(node)
    return jsonify({'message': 'Nodes registered', 'total_nodes': list(blockchain.nodes)}), 201


@app.route('/nodes/sync', methods=['GET'])
def sync():
    replaced = network.consensus()
    if replaced:
        return jsonify({'message': 'Chain replaced', 'new_chain': [vars(block) for block in blockchain.chain]}), 200
    return jsonify({'message': 'Chain is up to date', 'chain': [vars(block) for block in blockchain.chain]}), 200


@app.route('/')
def index():
    return app.send_static_file('index.html')


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=5000, type=int, help='port to listen on')
    args = parser.parse_args()

    # Now create the network with the correct port
    network = Network(blockchain, f'localhost:{args.port}')

    app.run(host='0.0.0.0', port=args.port)
