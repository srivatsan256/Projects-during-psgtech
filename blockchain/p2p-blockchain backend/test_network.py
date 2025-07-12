#!/usr/bin/env python3
"""
Test script for the P2P blockchain network
Demonstrates transactions, mining, and synchronization across all nodes
"""

import requests
import json
import time
import random

PORTS = [5000, 5001, 5002, 5003, 5004]

def test_node_connectivity():
    """Test if all nodes are online and connected"""
    print("Testing node connectivity...")
    for port in PORTS:
        try:
            response = requests.get(f'http://localhost:{port}/api/stats')
            if response.status_code == 200:
                stats = response.json()
                print(f"✓ Node {port}: {stats['total_nodes']} peers, {stats['total_blocks']} blocks")
            else:
                print(f"✗ Node {port}: HTTP {response.status_code}")
        except requests.RequestException as e:
            print(f"✗ Node {port}: Connection failed - {e}")
    print()

def create_test_transaction(sender_port, recipient_port):
    """Create a test transaction between two nodes"""
    try:
        transaction_data = {
            'sender': f'node_{sender_port}',
            'recipient': f'node_{recipient_port}',
            'amount': random.randint(1, 100)
        }
        
        response = requests.post(f'http://localhost:{sender_port}/api/transaction/new', 
                               json=transaction_data)
        if response.status_code == 201:
            print(f"✓ Transaction created: {transaction_data['sender']} -> {transaction_data['recipient']} ({transaction_data['amount']} coins)")
            return True
        else:
            print(f"✗ Transaction failed: {response.text}")
            return False
    except requests.RequestException as e:
        print(f"✗ Transaction failed: {e}")
        return False

def mine_block(port):
    """Mine a block on a specific node"""
    try:
        response = requests.get(f'http://localhost:{port}/api/mine')
        if response.status_code == 200:
            block_data = response.json()
            print(f"✓ Block mined on node {port}: Block #{block_data['index']}")
            return True
        else:
            print(f"✗ Mining failed on node {port}: {response.text}")
            return False
    except requests.RequestException as e:
        print(f"✗ Mining failed on node {port}: {e}")
        return False

def sync_nodes():
    """Sync all nodes"""
    print("Syncing all nodes...")
    for port in PORTS:
        try:
            response = requests.get(f'http://localhost:{port}/api/nodes/sync')
            if response.status_code == 200:
                result = response.json()
                print(f"✓ Node {port}: {result['message']}")
            else:
                print(f"✗ Node {port}: Sync failed")
        except requests.RequestException as e:
            print(f"✗ Node {port}: Sync failed - {e}")
    print()

def get_chain_info(port):
    """Get chain information from a node"""
    try:
        response = requests.get(f'http://localhost:{port}/api/chain')
        if response.status_code == 200:
            chain_data = response.json()
            return chain_data['length'], len(chain_data['chain'][0]['transactions']) if chain_data['chain'] else 0
        else:
            return 0, 0
    except requests.RequestException:
        return 0, 0

def display_network_status():
    """Display comprehensive network status"""
    print("=" * 60)
    print("BLOCKCHAIN NETWORK STATUS")
    print("=" * 60)
    
    for port in PORTS:
        try:
            # Get stats
            stats_response = requests.get(f'http://localhost:{port}/api/stats')
            pending_response = requests.get(f'http://localhost:{port}/api/pending')
            
            if stats_response.status_code == 200:
                stats = stats_response.json()
                pending = pending_response.json() if pending_response.status_code == 200 else {'count': 0}
                
                print(f"Node {port}:")
                print(f"  - Status: ✓ Online")
                print(f"  - Blocks: {stats['total_blocks']}")
                print(f"  - Pending Transactions: {pending['count']}")
                print(f"  - Connected Peers: {stats['total_nodes']}")
                print(f"  - Difficulty: {stats['difficulty']}")
            else:
                print(f"Node {port}: ✗ Offline")
        except requests.RequestException:
            print(f"Node {port}: ✗ Connection failed")
        print()

def run_comprehensive_test():
    """Run a comprehensive test of the blockchain network"""
    print("Starting comprehensive blockchain network test...")
    print("=" * 60)
    
    # Test 1: Check connectivity
    print("1. Testing node connectivity...")
    test_node_connectivity()
    
    # Test 2: Create multiple transactions
    print("2. Creating test transactions...")
    for i in range(3):
        sender = random.choice(PORTS)
        recipient = random.choice([p for p in PORTS if p != sender])
        create_test_transaction(sender, recipient)
    print()
    
    # Test 3: Mine blocks on different nodes
    print("3. Mining blocks on different nodes...")
    for port in random.sample(PORTS, 2):
        mine_block(port)
    print()
    
    # Test 4: Sync network
    print("4. Synchronizing network...")
    sync_nodes()
    
    # Test 5: Verify chain consistency
    print("5. Verifying chain consistency...")
    chain_lengths = []
    for port in PORTS:
        length, _ = get_chain_info(port)
        chain_lengths.append(length)
    
    if len(set(chain_lengths)) == 1:
        print(f"✓ All nodes have consistent chain length: {chain_lengths[0]} blocks")
    else:
        print(f"✗ Chain length inconsistency: {chain_lengths}")
    print()
    
    # Test 6: Display final status
    print("6. Final network status:")
    display_network_status()

def interactive_menu():
    """Interactive menu for manual testing"""
    while True:
        print("\n" + "=" * 40)
        print("BLOCKCHAIN NETWORK TEST MENU")
        print("=" * 40)
        print("1. Test node connectivity")
        print("2. Create transaction")
        print("3. Mine block")
        print("4. Sync network")
        print("5. Display network status")
        print("6. Run comprehensive test")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == '1':
            test_node_connectivity()
        elif choice == '2':
            try:
                sender = int(input("Enter sender port: "))
                recipient = int(input("Enter recipient port: "))
                if sender in PORTS and recipient in PORTS:
                    create_test_transaction(sender, recipient)
                else:
                    print("Invalid port numbers!")
            except ValueError:
                print("Please enter valid port numbers!")
        elif choice == '3':
            try:
                port = int(input("Enter port to mine on: "))
                if port in PORTS:
                    mine_block(port)
                else:
                    print("Invalid port number!")
            except ValueError:
                print("Please enter a valid port number!")
        elif choice == '4':
            sync_nodes()
        elif choice == '5':
            display_network_status()
        elif choice == '6':
            run_comprehensive_test()
        elif choice == '7':
            print("Goodbye!")
            break
        else:
            print("Invalid choice! Please try again.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        run_comprehensive_test()
    else:
        interactive_menu()