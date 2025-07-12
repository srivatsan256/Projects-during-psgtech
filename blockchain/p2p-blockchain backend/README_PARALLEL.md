# Parallel Blockchain Network

A peer-to-peer blockchain network that runs in parallel across multiple ports (5000, 5001, 5002, 5003, 5004).

## Overview

This blockchain implementation features:
- ✅ **Parallel Operation**: All 5 nodes run simultaneously and communicate with each other
- ✅ **Automatic Peer Discovery**: Nodes automatically connect to each other
- ✅ **Consensus Mechanism**: Longest valid chain wins
- ✅ **Transaction Broadcasting**: Transactions are propagated across all nodes
- ✅ **Block Broadcasting**: Mined blocks are shared with all peers
- ✅ **Thread-Safe Operations**: All operations are thread-safe for parallel execution
- ✅ **Real-time Synchronization**: Nodes sync periodically to maintain consistency

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Network

```bash
python start_network.py
```

This will:
- Start 5 blockchain nodes on ports 5000-5004
- Connect all nodes to each other automatically
- Display real-time network status
- Create some test transactions

### 3. Run Demo (in another terminal)

```bash
python demo_parallel.py
```

This demonstrates:
- Parallel transaction creation
- Parallel mining across multiple nodes
- Network synchronization
- Stress testing with concurrent operations

## Network Architecture

```
    Node 5000 ←→ Node 5001 ←→ Node 5002
        ↕           ↕           ↕
    Node 5004 ←→ Node 5003 ←→ [All Connected]
```

Each node:
- Runs independently on its own port
- Maintains its own copy of the blockchain
- Communicates with all other nodes
- Participates in consensus

## API Endpoints

Each node exposes the following API endpoints:

### Transaction Operations
- `POST /api/transaction/new` - Create a new transaction
- `GET /api/pending` - Get pending transactions

### Mining Operations
- `GET /api/mine` - Mine a new block

### Blockchain Operations
- `GET /api/chain` - Get the full blockchain
- `GET /api/validate` - Validate the blockchain

### Network Operations
- `POST /api/nodes/register` - Register peer nodes
- `GET /api/nodes/sync` - Synchronize with peers
- `GET /api/network/status` - Get network status

### Statistics
- `GET /api/stats` - Get node statistics

## Usage Examples

### Create a Transaction

```bash
curl -X POST http://localhost:5000/api/transaction/new \
  -H "Content-Type: application/json" \
  -d '{"sender": "Alice", "recipient": "Bob", "amount": 50}'
```

### Mine a Block

```bash
curl http://localhost:5001/api/mine
```

### Get Blockchain

```bash
curl http://localhost:5002/api/chain
```

### Check Network Status

```bash
curl http://localhost:5003/api/network/status
```

### Get Node Statistics

```bash
curl http://localhost:5004/api/stats
```

## Scripts

### start_network.py
- Starts all 5 nodes in parallel
- Automatically connects nodes to each other
- Provides real-time monitoring
- Handles graceful shutdown

### demo_parallel.py
- Demonstrates parallel blockchain operations
- Creates transactions across multiple nodes
- Shows mining on different nodes
- Tests network synchronization

### test_network.py
- Interactive testing tool
- Manual transaction creation
- Individual node testing
- Comprehensive network testing

## Parallel Features

### Transaction Broadcasting
When a transaction is created on any node, it's automatically broadcast to all other nodes:

```python
# Transaction created on Node 5000
# Automatically broadcast to Nodes 5001, 5002, 5003, 5004
```

### Parallel Mining
Multiple nodes can mine simultaneously:

```python
# Node 5000 mining Block #1
# Node 5002 mining Block #2
# Node 5004 mining Block #3
# Consensus determines the winning chain
```

### Automatic Synchronization
Nodes periodically sync to maintain consistency:

```python
# Every 5 seconds, each node checks peers for longer chains
# If a longer valid chain is found, it's adopted
```

## Architecture Details

### Thread Safety
All blockchain operations are protected by locks:
- Transaction creation
- Block mining
- Chain validation
- Network synchronization

### Consensus Algorithm
- **Longest Chain Rule**: The chain with the most blocks wins
- **Validation**: Only valid chains are accepted
- **Automatic Resolution**: Conflicts are resolved automatically

### Network Communication
- **HTTP REST API**: All inter-node communication uses HTTP
- **Parallel Requests**: Multiple requests are sent simultaneously
- **Timeout Handling**: Robust error handling for network issues

## Monitoring

The network provides comprehensive monitoring:

### Real-time Status
```
Node 5000: ✓ Online - 3 blocks, 2 pending, 4 peers
Node 5001: ✓ Online - 3 blocks, 1 pending, 4 peers
Node 5002: ✓ Online - 3 blocks, 0 pending, 4 peers
Node 5003: ✓ Online - 3 blocks, 1 pending, 4 peers
Node 5004: ✓ Online - 3 blocks, 0 pending, 4 peers
```

### Chain Consistency
```
✓ Chain consistency: All nodes have 3 blocks
```

### Network Health
```
Network Health: 5/5 nodes online
```

## Troubleshooting

### Port Already in Use
```bash
# Kill processes using the ports
sudo lsof -ti:5000-5004 | xargs kill -9
```

### Network Not Connecting
- Check firewall settings
- Ensure all nodes are running
- Verify network connectivity

### Chain Inconsistency
- Trigger manual sync: `curl http://localhost:5000/api/nodes/sync`
- Check network logs for errors
- Restart the network if necessary

## Advanced Usage

### Custom Port Configuration
Modify the `PORTS` list in the scripts:

```python
PORTS = [5000, 5001, 5002, 5003, 5004]  # Default
PORTS = [6000, 6001, 6002, 6003, 6004]  # Custom
```

### Adding More Nodes
1. Add ports to the `PORTS` list
2. Update the scripts accordingly
3. Restart the network

### Performance Tuning
- Adjust `sync_interval` for more/less frequent synchronization
- Modify `difficulty` for faster/slower mining
- Change `timeout` values for network requests

## Security Considerations

- **Local Network Only**: This implementation is designed for local testing
- **No Authentication**: API endpoints are open
- **No Encryption**: Communication is not encrypted
- **Development Use**: Not suitable for production

## Contributing

Feel free to contribute improvements:
- Enhanced consensus algorithms
- Better error handling
- Performance optimizations
- Security features

## License

This project is for educational purposes.