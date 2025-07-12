#!/usr/bin/env python3
"""
Parallel Blockchain Network Demo
Demonstrates the blockchain network working in parallel across all 5 ports
"""

import requests
import json
import time
import threading
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PORTS = [5000, 5001, 5002, 5003, 5004]
USERS = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry']

def create_transaction(sender_port, sender, recipient, amount):
    """Create a transaction on a specific node"""
    try:
        transaction_data = {
            'sender': sender,
            'recipient': recipient,
            'amount': amount
        }
        
        response = requests.post(f'http://localhost:{sender_port}/api/transaction/new', 
                               json=transaction_data, timeout=5)
        
        if response.status_code == 201:
            logger.info(f"✓ Transaction created on node {sender_port}: {sender} -> {recipient} ({amount} coins)")
            return True
        else:
            logger.error(f"✗ Transaction failed on node {sender_port}: {response.text}")
            return False
    except requests.RequestException as e:
        logger.error(f"✗ Transaction failed on node {sender_port}: {e}")
        return False

def mine_block(port):
    """Mine a block on a specific node"""
    try:
        response = requests.get(f'http://localhost:{port}/api/mine', timeout=30)
        if response.status_code == 200:
            block_data = response.json()
            logger.info(f"✓ Block #{block_data['index']} mined on node {port} with {len(block_data['transactions'])} transactions")
            return True
        else:
            logger.error(f"✗ Mining failed on node {port}: {response.text}")
            return False
    except requests.RequestException as e:
        logger.error(f"✗ Mining failed on node {port}: {e}")
        return False

def get_node_stats(port):
    """Get stats from a specific node"""
    try:
        response = requests.get(f'http://localhost:{port}/api/stats', timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.RequestException:
        return None

def wait_for_network():
    """Wait for the network to be ready"""
    logger.info("Waiting for network to be ready...")
    max_attempts = 30
    
    for attempt in range(max_attempts):
        ready_nodes = 0
        for port in PORTS:
            stats = get_node_stats(port)
            if stats and stats.get('total_nodes', 0) >= 4:  # Should have 4 peers
                ready_nodes += 1
        
        if ready_nodes == len(PORTS):
            logger.info("✓ All nodes are ready and connected!")
            return True
        
        logger.info(f"Network not ready: {ready_nodes}/{len(PORTS)} nodes connected. Attempt {attempt + 1}/{max_attempts}")
        time.sleep(2)
    
    logger.error("Network failed to initialize properly")
    return False

def display_network_status():
    """Display comprehensive network status"""
    logger.info("=" * 80)
    logger.info("BLOCKCHAIN NETWORK STATUS")
    logger.info("=" * 80)
    
    total_blocks = []
    total_pending = []
    
    for port in PORTS:
        stats = get_node_stats(port)
        if stats:
            logger.info(f"Node {port}: {stats['total_blocks']} blocks, "
                       f"{stats['pending_transactions']} pending, {stats['total_nodes']} peers")
            total_blocks.append(stats['total_blocks'])
            total_pending.append(stats['pending_transactions'])
        else:
            logger.error(f"Node {port}: ✗ Unreachable")
    
    # Check consistency
    if len(set(total_blocks)) == 1:
        logger.info(f"✓ Chain consistency: All nodes have {total_blocks[0]} blocks")
    else:
        logger.warning(f"✗ Chain inconsistency: {total_blocks}")
    
    logger.info("=" * 80)

def parallel_transaction_demo():
    """Demonstrate parallel transaction creation across all nodes"""
    logger.info("Starting parallel transaction demo...")
    
    # Create multiple transactions in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        
        # Create 20 random transactions
        for i in range(20):
            sender = random.choice(USERS)
            recipient = random.choice([u for u in USERS if u != sender])
            amount = random.randint(1, 100)
            sender_port = random.choice(PORTS)
            
            future = executor.submit(create_transaction, sender_port, sender, recipient, amount)
            futures.append(future)
        
        # Wait for all transactions to complete
        successful_transactions = 0
        for future in as_completed(futures):
            if future.result():
                successful_transactions += 1
        
        logger.info(f"✓ Created {successful_transactions}/20 transactions in parallel")

def parallel_mining_demo():
    """Demonstrate parallel mining across multiple nodes"""
    logger.info("Starting parallel mining demo...")
    
    # Check if there are pending transactions
    pending_count = 0
    for port in PORTS:
        stats = get_node_stats(port)
        if stats:
            pending_count += stats['pending_transactions']
    
    if pending_count == 0:
        logger.warning("No pending transactions to mine")
        return
    
    # Mine blocks in parallel on different nodes
    with ThreadPoolExecutor(max_workers=3) as executor:
        mining_ports = random.sample(PORTS, 3)  # Mine on 3 random nodes
        futures = []
        
        for port in mining_ports:
            future = executor.submit(mine_block, port)
            futures.append((port, future))
        
        # Wait for mining to complete
        successful_mines = 0
        for port, future in futures:
            if future.result():
                successful_mines += 1
        
        logger.info(f"✓ Successfully mined {successful_mines}/3 blocks in parallel")

def stress_test():
    """Perform a stress test with many parallel operations"""
    logger.info("Starting stress test...")
    
    def stress_worker(worker_id):
        """Worker function for stress testing"""
        logger.info(f"Stress worker {worker_id} started")
        
        for i in range(10):
            # Create a transaction
            sender = f"Worker_{worker_id}"
            recipient = random.choice(USERS)
            amount = random.randint(1, 50)
            port = random.choice(PORTS)
            
            create_transaction(port, sender, recipient, amount)
            time.sleep(0.1)  # Small delay
        
        logger.info(f"Stress worker {worker_id} completed")
    
    # Start multiple stress workers
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i in range(5):
            future = executor.submit(stress_worker, i)
            futures.append(future)
        
        # Wait for all workers to complete
        for future in as_completed(futures):
            future.result()
    
    logger.info("✓ Stress test completed")

def sync_test():
    """Test network synchronization"""
    logger.info("Testing network synchronization...")
    
    # Trigger sync on all nodes in parallel
    with ThreadPoolExecutor(max_workers=len(PORTS)) as executor:
        futures = []
        
        for port in PORTS:
            future = executor.submit(lambda p: requests.get(f'http://localhost:{p}/api/nodes/sync', timeout=10), port)
            futures.append((port, future))
        
        # Wait for all syncs to complete
        for port, future in futures:
            try:
                response = future.result()
                if response.status_code == 200:
                    logger.info(f"✓ Sync successful on node {port}")
                else:
                    logger.error(f"✗ Sync failed on node {port}")
            except Exception as e:
                logger.error(f"✗ Sync error on node {port}: {e}")
    
    logger.info("✓ Network synchronization test completed")

def main():
    """Main demonstration function"""
    logger.info("Starting Parallel Blockchain Network Demo")
    logger.info("=" * 80)
    
    # Wait for network to be ready
    if not wait_for_network():
        logger.error("Network not ready, exiting demo")
        return
    
    # Display initial status
    display_network_status()
    
    # Demo 1: Parallel transaction creation
    logger.info("\n🔥 DEMO 1: Parallel Transaction Creation")
    parallel_transaction_demo()
    time.sleep(3)
    display_network_status()
    
    # Demo 2: Parallel mining
    logger.info("\n🔥 DEMO 2: Parallel Mining")
    parallel_mining_demo()
    time.sleep(5)  # Wait for mining to complete
    display_network_status()
    
    # Demo 3: Network synchronization
    logger.info("\n🔥 DEMO 3: Network Synchronization")
    sync_test()
    time.sleep(3)
    display_network_status()
    
    # Demo 4: Stress test
    logger.info("\n🔥 DEMO 4: Stress Test")
    stress_test()
    time.sleep(3)
    display_network_status()
    
    # Final mining to process all pending transactions
    logger.info("\n🔥 FINAL: Processing all pending transactions")
    parallel_mining_demo()
    time.sleep(5)
    display_network_status()
    
    logger.info("=" * 80)
    logger.info("✅ Parallel Blockchain Network Demo Completed!")
    logger.info("=" * 80)
    
    # Show final statistics
    all_stats = []
    for port in PORTS:
        stats = get_node_stats(port)
        if stats:
            all_stats.append(stats)
    
    if all_stats:
        total_blocks = all_stats[0]['total_blocks']
        total_pending = sum(s['pending_transactions'] for s in all_stats)
        logger.info(f"Final Network State:")
        logger.info(f"  - Total blocks: {total_blocks}")
        logger.info(f"  - Pending transactions: {total_pending}")
        logger.info(f"  - Network consistency: {'✓' if len(set(s['total_blocks'] for s in all_stats)) == 1 else '✗'}")

if __name__ == "__main__":
    main()