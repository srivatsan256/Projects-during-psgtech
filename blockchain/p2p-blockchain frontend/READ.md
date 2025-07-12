P2P Blockchain Network Simulation
Overview
This project simulates a peer-to-peer blockchain network with multiple nodes, each maintaining an independent blockchain copy. Nodes communicate via HTTP to register, sync, and resolve conflicts using a longest-chain-wins consensus algorithm.
Requirements

Python 3.8+
Flask
requests

Setup

Clone the repository:git clone <repository-url>
cd p2p-blockchain


Create and activate a virtual environment:python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:pip install -r requirements.txt


Run multiple nodes on different ports:python app.py --port 5000
python app.py --port 5001
python app.py --port 5002



Usage

Access a node at http://localhost:<port> (e.g., http://localhost:5000).
Use the web interface to:
Add transactions
Mine blocks
Register other nodes (e.g., localhost:5001)


Sync nodes using the /nodes/sync endpoint to resolve conflicts.

Endpoints

POST /transaction/new: Add a new transaction
GET /mine: Mine a new block
GET /chain: Get the full blockchain
POST /nodes/register: Register peer nodes
GET /nodes/sync: Sync with peers

Frontend Integration
The provided index.html is a basic template. To integrate with a bolt.new frontend:

Generate a modern UI using bolt.new.
Replace templates/index.html and static/style.css with bolt.new's assets.
Ensure the JavaScript functions (addTransaction, mineBlock, registerNode) align with the new UI components.

Running the Simulation

Start multiple nodes on different ports.
Register nodes with each other using the web interface or curl:curl -X POST -H "Content-Type: application/json" -d '{"nodes": ["localhost:5001", "localhost:5002"]}' http://localhost:5000/nodes/register


Add transactions and mine blocks via the web interface.
Sync nodes to observe the consensus algorithm in action.

Screenshots/Logs

Save logs from the Flask console to demonstrate node syncing and conflict resolution.
Take screenshots of the web interface showing the blockchain state.

Bonus Features

Basic HTML frontend included
Transaction validation can be added to blockchain.py
Node status dashboard available via the /chain endpoint
