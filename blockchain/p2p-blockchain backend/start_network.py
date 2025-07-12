#!/usr/bin/env python3
"""
Multi-node blockchain network starter
Launches blockchain nodes on ports 5000, 5001, 5002, 5003, 5004
Enhanced for parallel operation and better coordination
"""

import subprocess
import time
import requests
import json
import os
import signal
import sys
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
PORTS = [5000, 5001, 5002, 5003, 5004]
PROCESSES = []
MAX_STARTUP_TIME = 60  # seconds
HEALTH_CHECK_INTERVAL = 5  # seconds

def start_node(port):
    """Start a single blockchain node"""
    logger.info(f"Starting node on port {port}...")
    try:
        process = subprocess.Popen([
            sys.executable, 'app.py', '--port', str(port)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        PROCESSES.append(process)
        return process
    except Exception as e:
        logger.error(f"Failed to start node on port {port}: {e}")
        return None

def wait_for_node(port, timeout=30):
    """Wait for a node to be ready with health checks"""
    logger.info(f"Waiting for node on port {port} to be ready...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f'http://localhost:{port}/api/stats', timeout=2)
            if response.status_code == 200:
                logger.info(f"Node on port {port} is ready!")
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    
    logger.error(f"Node on port {port} failed to start within {timeout} seconds")
    return False

def register_peers_parallel(port, peer_ports):
    """Register peer nodes with a specific node"""
    try:
        peers = [f'localhost:{peer_port}' for peer_port in peer_ports if peer_port != port]
        if peers:
            logger.info(f"Registering {len(peers)} peers for node {port}")
            response = requests.post(f'http://localhost:{port}/api/nodes/register', 
                                   json={'nodes': peers}, timeout=10)
            if response.status_code == 201:
                result = response.json()
                logger.info(f"Node {port}: Successfully registered peers - {result.get('total_nodes', 0)} total nodes")
                return True
            else:
                logger.error(f"Failed to register peers for node {port}: {response.text}")
                return False
    except requests.RequestException as e:
        logger.error(f"Error registering peers for node {port}: {e}")
        return False

def check_node_health(port):
    """Check if a node is healthy"""
    try:
        response = requests.get(f'http://localhost:{port}/api/stats', timeout=2)
        if response.status_code == 200:
            stats = response.json()
            return {
                'port': port,
                'healthy': True,
                'blocks': stats.get('total_blocks', 0),
                'pending': stats.get('pending_transactions', 0),
                'peers': stats.get('total_nodes', 0)
            }
        else:
            return {'port': port, 'healthy': False, 'error': f'HTTP {response.status_code}'}
    except requests.RequestException as e:
        return {'port': port, 'healthy': False, 'error': str(e)}

def setup_network():
    """Setup the complete network with parallel operations"""
    logger.info("Setting up blockchain network...")
    
    # Step 1: Start all nodes in parallel
    logger.info("Starting all nodes in parallel...")
    with ThreadPoolExecutor(max_workers=len(PORTS)) as executor:
        start_futures = {executor.submit(start_node, port): port for port in PORTS}
        
        for future in as_completed(start_futures):
            port = start_futures[future]
            try:
                process = future.result()
                if process:
                    logger.info(f"Node {port} started successfully")
                else:
                    logger.error(f"Failed to start node {port}")
            except Exception as e:
                logger.error(f"Exception starting node {port}: {e}")
    
    # Small delay to let processes initialize
    time.sleep(3)
    
    # Step 2: Wait for all nodes to be ready in parallel
    logger.info("Waiting for all nodes to be ready...")
    with ThreadPoolExecutor(max_workers=len(PORTS)) as executor:
        ready_futures = {executor.submit(wait_for_node, port): port for port in PORTS}
        
        all_ready = True
        for future in as_completed(ready_futures):
            port = ready_futures[future]
            try:
                ready = future.result()
                if not ready:
                    logger.error(f"Node {port} failed to start!")
                    all_ready = False
            except Exception as e:
                logger.error(f"Exception waiting for node {port}: {e}")
                all_ready = False
    
    if not all_ready:
        logger.error("Not all nodes started successfully!")
        return False
    
    # Step 3: Register peers for each node in parallel
    logger.info("Connecting nodes to each other...")
    with ThreadPoolExecutor(max_workers=len(PORTS)) as executor:
        peer_futures = {executor.submit(register_peers_parallel, port, PORTS): port for port in PORTS}
        
        for future in as_completed(peer_futures):
            port = peer_futures[future]
            try:
                success = future.result()
                if success:
                    logger.info(f"Peer registration successful for node {port}")
                else:
                    logger.warning(f"Peer registration failed for node {port}")
            except Exception as e:
                logger.error(f"Exception registering peers for node {port}: {e}")
    
    # Step 4: Final health check
    logger.info("Performing final health check...")
    time.sleep(2)  # Allow time for peer connections to establish
    
    with ThreadPoolExecutor(max_workers=len(PORTS)) as executor:
        health_futures = {executor.submit(check_node_health, port): port for port in PORTS}
        
        healthy_nodes = 0
        for future in as_completed(health_futures):
            port = health_futures[future]
            try:
                health = future.result()
                if health['healthy']:
                    logger.info(f"Node {port}: ✓ Healthy - {health['blocks']} blocks, {health['peers']} peers")
                    healthy_nodes += 1
                else:
                    logger.error(f"Node {port}: ✗ Unhealthy - {health.get('error', 'Unknown error')}")
            except Exception as e:
                logger.error(f"Exception checking health for node {port}: {e}")
    
    if healthy_nodes == len(PORTS):
        logger.info("Network setup complete! All nodes are healthy.")
        return True
    else:
        logger.warning(f"Network setup completed with {healthy_nodes}/{len(PORTS)} healthy nodes")
        return healthy_nodes > 0

def shutdown_network():
    """Shutdown all nodes gracefully"""
    logger.info("Shutting down network...")
    
    # Try graceful shutdown first
    for process in PROCESSES:
        try:
            process.terminate()
        except Exception as e:
            logger.error(f"Error terminating process: {e}")
    
    # Wait for processes to terminate
    time.sleep(5)
    
    # Force kill if necessary
    for process in PROCESSES:
        try:
            if process.poll() is None:  # Still running
                process.kill()
                logger.warning("Force-killed a stubborn process")
        except Exception as e:
            logger.error(f"Error killing process: {e}")
    
    logger.info("Network shutdown complete!")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    shutdown_network()
    sys.exit(0)

def monitor_nodes():
    """Monitor node status with enhanced reporting"""
    while True:
        try:
            logger.info("=" * 60)
            logger.info("BLOCKCHAIN NETWORK STATUS")
            logger.info("=" * 60)
            
            # Check all nodes in parallel
            with ThreadPoolExecutor(max_workers=len(PORTS)) as executor:
                health_futures = {executor.submit(check_node_health, port): port for port in PORTS}
                
                healthy_count = 0
                total_blocks = set()
                total_peers = set()
                
                for future in as_completed(health_futures):
                    port = health_futures[future]
                    try:
                        health = future.result()
                        if health['healthy']:
                            logger.info(f"Node {port}: ✓ Online - {health['blocks']} blocks, "
                                      f"{health['pending']} pending, {health['peers']} peers")
                            healthy_count += 1
                            total_blocks.add(health['blocks'])
                            total_peers.add(health['peers'])
                        else:
                            logger.error(f"Node {port}: ✗ Offline - {health.get('error', 'Unknown error')}")
                    except Exception as e:
                        logger.error(f"Node {port}: ✗ Error - {e}")
            
            # Network health summary
            logger.info("=" * 60)
            logger.info(f"Network Health: {healthy_count}/{len(PORTS)} nodes online")
            if len(total_blocks) > 1:
                logger.warning(f"Chain inconsistency detected: {sorted(total_blocks)} blocks across nodes")
            else:
                logger.info(f"Chain consistency: {list(total_blocks)[0] if total_blocks else 0} blocks")
            
            logger.info("Available endpoints:")
            for port in PORTS:
                logger.info(f"  http://localhost:{port} - Node {port}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
        
        time.sleep(15)  # Monitor every 15 seconds

def create_test_transactions():
    """Create some test transactions to demonstrate the network"""
    logger.info("Creating test transactions...")
    
    test_transactions = [
        {'sender': 'Alice', 'recipient': 'Bob', 'amount': 50},
        {'sender': 'Bob', 'recipient': 'Charlie', 'amount': 25},
        {'sender': 'Charlie', 'recipient': 'Alice', 'amount': 10},
    ]
    
    for i, tx in enumerate(test_transactions):
        try:
            port = PORTS[i % len(PORTS)]
            response = requests.post(f'http://localhost:{port}/api/transaction/new', 
                                   json=tx, timeout=5)
            if response.status_code == 201:
                logger.info(f"Test transaction created on node {port}: {tx}")
            else:
                logger.error(f"Failed to create test transaction on node {port}")
        except Exception as e:
            logger.error(f"Error creating test transaction: {e}")
        
        time.sleep(1)

def main():
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting P2P Blockchain Network")
    logger.info(f"Ports: {', '.join(map(str, PORTS))}")
    logger.info("-" * 60)
    
    if setup_network():
        logger.info("Blockchain network is running!")
        logger.info("Press Ctrl+C to stop all nodes")
        
        # Create some test transactions
        time.sleep(5)
        create_test_transactions()
        
        # Start monitoring in a separate thread
        monitor_thread = Thread(target=monitor_nodes, daemon=True)
        monitor_thread.start()
        
        # Keep main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
    else:
        logger.error("Failed to setup network!")
        shutdown_network()
        sys.exit(1)

if __name__ == "__main__":
    main()