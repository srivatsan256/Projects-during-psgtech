# Parallel Blockchain Network Backend

A peer-to-peer blockchain network that runs in parallel across multiple ports (5000, 5001, 5002, 5003, 5004).

## Features
- Parallel operation: 5 nodes run simultaneously and communicate
- Automatic peer discovery
- Longest-chain-wins consensus
- Transaction and block broadcasting
- Thread-safe, real-time synchronization

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Network

```bash
python start_network.py
```
- Starts 5 blockchain nodes on ports 5000-5004
- Nodes connect automatically and display real-time status

### 3. Run Demo (Optional, in another terminal)

```bash
python demo_parallel.py
```
- Demonstrates parallel transaction creation, mining, and network sync

## API Endpoints

Each node exposes the following endpoints:
- `POST /api/transaction/new` — Create a new transaction
- `GET /api/pending` — Get pending transactions
- `GET /api/mine` — Mine a new block
- `GET /api/chain` — Get the full blockchain
- `GET /api/validate` — Validate the blockchain
- `POST /api/nodes/register` — Register peer nodes
- `GET /api/nodes/sync` — Synchronize with peers
- `GET /api/network/status` — Get network status
- `GET /api/stats` — Get node statistics

## Usage Examples

**Create a Transaction:**
```bash
curl -X POST http://localhost:5000/api/transaction/new \
  -H "Content-Type: application/json" \
  -d '{"sender": "Alice", "recipient": "Bob", "amount": 50}'
```

**Mine a Block:**
```bash
curl http://localhost:5001/api/mine
```

**Get Blockchain:**
```bash
curl http://localhost:5002/api/chain
```

**Check Network Status:**
```bash
curl http://localhost:5003/api/network/status
```

**Get Node Statistics:**
```bash
curl http://localhost:5004/api/stats
```

## Scripts
- `start_network.py`: Starts all 5 nodes in parallel
- `demo_parallel.py`: Demonstrates parallel blockchain operations
- `test_network.py`: Interactive/manual network testing

## Architecture
- Each node runs independently on its own port
- Nodes maintain their own blockchain copy and communicate via HTTP REST API
- Consensus: Longest valid chain wins
- Thread safety: All operations are protected by locks
- Real-time synchronization: Nodes sync every 5 seconds

## Troubleshooting
- **Port in use:**
  ```bash
  sudo lsof -ti:5000-5004 | xargs kill -9
  ```
- **Network not connecting:**
  - Check firewall, ensure all nodes are running
- **Chain inconsistency:**
  - Trigger manual sync: `curl http://localhost:5000/api/nodes/sync`
  - Check logs, restart network if needed

## Advanced Usage
- **Custom ports:** Edit the `PORTS` list in the scripts
- **Add more nodes:** Add ports to `PORTS`, update scripts, restart
- **Performance tuning:** Adjust `sync_interval`, mining `difficulty`, or network `timeout`

## Security
- For local testing only; no authentication on API endpoints

---

For more details, see `README_PARALLEL.md` in this directory.