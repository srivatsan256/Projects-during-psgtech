#!/usr/bin/env python3
"""
Multi-node blockchain network starter
Launches blockchain nodes on ports 5000, 5001, 5002, 5003, 5004
"""

import subprocess
import time
import requests
import json
import os
import signal
import sys
from threading import Thread

# Configuration
PORTS = [5000, 5001, 5002, 5003, 5004]
PROCESSES = []

def start_node(port):
    """Start a single blockchain node"""
    print(f"Starting node on port {port}...")
    process = subprocess.Popen([
        sys.executable, 'app.py', '--port', str(port)
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    PROCESSES.append(process)
    return process

def wait_for_node(port, timeout=30):
    """Wait for a node to be ready"""
    print(f"Waiting for node on port {port} to be ready...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f'http://localhost:{port}/api/stats', timeout=1)
            if response.status_code == 200:
                print(f"Node on port {port} is ready!")
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False

def register_peers(port, peer_ports):
    """Register peer nodes with a specific node"""
    try:
        peers = [f'localhost:{peer_port}' for peer_port in peer_ports if peer_port != port]
        if peers:
            print(f"Registering peers for node {port}: {peers}")
            response = requests.post(f'http://localhost:{port}/api/nodes/register', 
                                   json={'nodes': peers})
            if response.status_code == 201:
                print(f"Successfully registered {len(peers)} peers for node {port}")
            else:
                print(f"Failed to register peers for node {port}: {response.text}")
    except requests.RequestException as e:
        print(f"Error registering peers for node {port}: {e}")

def setup_network():
    """Setup the complete network"""
    print("Setting up blockchain network...")
    
    # Start all nodes
    for port in PORTS:
        start_node(port)
        time.sleep(2)  # Small delay between starts
    
    # Wait for all nodes to be ready
    print("Waiting for all nodes to be ready...")
    for port in PORTS:
        if not wait_for_node(port):
            print(f"Node on port {port} failed to start!")
            return False
    
    # Register peers for each node
    print("Connecting nodes to each other...")
    for port in PORTS:
        register_peers(port, PORTS)
    
    print("Network setup complete!")
    return True

def shutdown_network():
    """Shutdown all nodes"""
    print("Shutting down network...")
    for process in PROCESSES:
        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    print("Network shutdown complete!")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print(f"\nReceived signal {signum}, shutting down...")
    shutdown_network()
    sys.exit(0)

def monitor_nodes():
    """Monitor node status"""
    while True:
        print("\n" + "="*50)
        print("BLOCKCHAIN NETWORK STATUS")
        print("="*50)
        
        for port in PORTS:
            try:
                response = requests.get(f'http://localhost:{port}/api/stats', timeout=2)
                if response.status_code == 200:
                    stats = response.json()
                    print(f"Node {port}: ✓ Online - {stats['total_blocks']} blocks, "
                          f"{stats['pending_transactions']} pending, {stats['total_nodes']} peers")
                else:
                    print(f"Node {port}: ✗ Error - HTTP {response.status_code}")
            except requests.RequestException:
                print(f"Node {port}: ✗ Offline")
        
        print("="*50)
        print("Available endpoints:")
        for port in PORTS:
            print(f"  http://localhost:{port} - Node {port}")
        print("="*50)
        
        time.sleep(10)

def main():
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("Starting P2P Blockchain Network")
    print(f"Ports: {', '.join(map(str, PORTS))}")
    print("-" * 50)
    
    if setup_network():
        print("\nBlockchain network is running!")
        print("Press Ctrl+C to stop all nodes")
        
        # Start monitoring in a separate thread
        monitor_thread = Thread(target=monitor_nodes, daemon=True)
        monitor_thread.start()
        
        # Keep main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        print("Failed to setup network!")
        shutdown_network()
        sys.exit(1)

if __name__ == "__main__":
    main()