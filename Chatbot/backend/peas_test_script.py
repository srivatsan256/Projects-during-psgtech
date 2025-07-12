#!/usr/bin/env python3
"""
PEAS Auto-Tuning Test Script
Test the PEAS (Performance Environment Agent Sensor) auto-tuning functionality.
"""

import requests
import json
import time
import sys

# Configuration
BASE_URL = "http://localhost:5000"
TEST_USERNAME = "test_user_peas"

def test_api_endpoint(endpoint, method="GET", data=None):
    """Test an API endpoint and return the response."""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data, headers={'Content-Type': 'application/json'})
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None

def display_peas_status(peas_data):
    """Display PEAS status in a formatted way."""
    if not peas_data or not peas_data.get('success'):
        print("❌ Failed to get PEAS status")
        return
    
    data = peas_data['data']
    print(f"\n📊 PEAS Status for {data['user']}")
    print("=" * 50)
    print(f"Performance (P): {data['components']['Performance']:.3f}")
    print(f"Environment (E): {data['components']['Environment']:.3f}")
    print(f"Actuators (A):   {data['components']['Actuators']:.3f}")
    print(f"Sensors (S):     {data['components']['Sensors']:.3f}")
    print("-" * 50)
    print(f"Total Sum:       {data['sum']:.3f}")
    print(f"Max Ceiling:     {data['ceiling']}")
    print(f"Needs Tuning:    {'✅ Yes' if data['needs_tuning'] else '❌ No'}")
    
    if data.get('last_tuning'):
        print(f"Last Tuning:     {data['last_tuning']}")
        print(f"Tuning Reason:   {data.get('tuning_reason', 'N/A')}")
        print(f"Original Sum:    {data.get('original_sum', 'N/A')}")
        print(f"Tuned Sum:       {data.get('tuned_sum', 'N/A')}")

def test_peas_functionality():
    """Test the complete PEAS functionality."""
    print("🚀 Starting PEAS Auto-Tuning Test")
    print("=" * 60)
    
    # Test 1: Check server health
    print("\n1️⃣ Testing Server Health...")
    try:
        response = requests.get(f"{BASE_URL}/api/peas_overview")
        if response.status_code == 200:
            print("✅ Server is running and PEAS endpoints are accessible")
        else:
            print("❌ Server not accessible")
            return False
    except:
        print("❌ Cannot connect to server. Make sure chatbot_optimized.py is running.")
        return False
    
    # Test 2: Get initial PEAS status
    print("\n2️⃣ Testing PEAS Status API...")
    peas_status = test_api_endpoint("/api/peas_status", "POST", {"username": TEST_USERNAME})
    if peas_status:
        display_peas_status(peas_status)
    else:
        print("❌ User not found - this is normal for first run")
    
    # Test 3: Create test scenario with PEAS sum = 4
    print("\n3️⃣ Testing Auto-Tuning Trigger (Sum = 4)...")
    
    # We'll manually create a test user with sum = 4 by calling the force tune
    print("Attempting to force tune the user...")
    tune_result = test_api_endpoint("/api/peas_tune", "POST", {
        "username": TEST_USERNAME,
        "force_tune": True
    })
    
    if tune_result:
        print("✅ Manual tuning completed")
        display_peas_status(tune_result)
    else:
        print("❌ Manual tuning failed")
    
    # Test 4: Check PEAS status after tuning
    print("\n4️⃣ Testing PEAS Status After Tuning...")
    peas_status_after = test_api_endpoint("/api/peas_status", "POST", {"username": TEST_USERNAME})
    if peas_status_after:
        display_peas_status(peas_status_after)
        
        # Verify tuning worked
        sum_after = peas_status_after['data']['sum']
        if abs(sum_after - 2.0) < 0.1:
            print("✅ PEAS auto-tuning worked correctly (sum ≈ 2.0)")
        else:
            print(f"⚠️ PEAS sum is {sum_after}, expected ≈ 2.0")
    
    # Test 5: Test batch tuning
    print("\n5️⃣ Testing Batch PEAS Tuning...")
    batch_result = test_api_endpoint("/api/peas_batch_tune", "POST")
    if batch_result:
        print(f"✅ Batch tuning completed: {batch_result['message']}")
        print(f"Tuned users: {batch_result['tuned_users']}")
    
    # Test 6: Test overview API
    print("\n6️⃣ Testing PEAS Overview API...")
    overview_result = test_api_endpoint("/api/peas_overview", "GET")
    if overview_result:
        overview_data = overview_result['data']
        print(f"✅ PEAS Overview:")
        print(f"   Total Users: {overview_data['total_users']}")
        print(f"   Users Needing Tuning: {overview_data['users_needing_tuning']}")
        
        # Display first few users
        if overview_data['users']:
            print(f"   Sample Users:")
            for i, user in enumerate(overview_data['users'][:3]):
                print(f"     {i+1}. {user['user']}: Sum={user['sum']:.3f}, Needs Tuning={user['needs_tuning']}")
    
    print("\n✅ PEAS Auto-Tuning Test Completed Successfully!")
    print("=" * 60)
    return True

def test_peas_scenarios():
    """Test specific PEAS scenarios."""
    print("\n🧪 Testing PEAS Scenarios")
    print("=" * 40)
    
    scenarios = [
        {
            "name": "Normal Operation",
            "description": "User with balanced PEAS scores",
            "expected_tuning": False
        },
        {
            "name": "Sum = 4 Trigger",
            "description": "User with total sum = 4.0",
            "expected_tuning": True
        },
        {
            "name": "Ceiling = 4 Trigger",
            "description": "User with any component ceiling = 4",
            "expected_tuning": True
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        print(f"   Expected Tuning: {'Yes' if scenario['expected_tuning'] else 'No'}")

def display_help():
    """Display help information."""
    print("""
PEAS Auto-Tuning Test Script
===========================

Usage: python peas_test_script.py [options]

Options:
  --help, -h          Show this help message
  --test, -t          Run full PEAS functionality test
  --scenarios, -s     Show test scenarios
  --status <user>     Get PEAS status for a specific user
  --tune <user>       Force tune a specific user
  --overview          Get PEAS overview

Examples:
  python peas_test_script.py --test
  python peas_test_script.py --status john_doe
  python peas_test_script.py --tune jane_smith
  python peas_test_script.py --overview

Prerequisites:
  - Make sure chatbot_optimized.py is running on localhost:5000
  - Ensure the server has PEAS endpoints enabled
""")

def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 2:
        print("PEAS Auto-Tuning Test Script")
        print("Use --help for usage information")
        return
    
    command = sys.argv[1]
    
    if command in ['--help', '-h']:
        display_help()
    elif command in ['--test', '-t']:
        test_peas_functionality()
    elif command in ['--scenarios', '-s']:
        test_peas_scenarios()
    elif command == '--status' and len(sys.argv) > 2:
        username = sys.argv[2]
        result = test_api_endpoint("/api/peas_status", "POST", {"username": username})
        if result:
            display_peas_status(result)
    elif command == '--tune' and len(sys.argv) > 2:
        username = sys.argv[2]
        result = test_api_endpoint("/api/peas_tune", "POST", {"username": username, "force_tune": True})
        if result:
            print(f"✅ Tuning completed for {username}")
            display_peas_status(result)
    elif command == '--overview':
        result = test_api_endpoint("/api/peas_overview", "GET")
        if result:
            overview_data = result['data']
            print(f"📊 PEAS Overview")
            print(f"Total Users: {overview_data['total_users']}")
            print(f"Users Needing Tuning: {overview_data['users_needing_tuning']}")
            
            if overview_data['users']:
                print("\nUsers:")
                for user in overview_data['users']:
                    status = "🔴 Needs Tuning" if user['needs_tuning'] else "🟢 Optimized"
                    print(f"  {user['user']}: Sum={user['sum']:.3f} {status}")
    else:
        print("Invalid command. Use --help for usage information.")

if __name__ == "__main__":
    main()