#!/usr/bin/env python3

import os
import subprocess
import sys

def run_command(command, cwd=None):
    """Run a command and return its output."""
    try:
        result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running command: {command}")
            print(f"Error output: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Exception running command: {command}")
        print(f"Exception: {e}")
        return False

def setup_backend():
    """Set up the Flask backend."""
    print("Setting up Flask backend...")
    backend_dir = "p2p-blockchain backend"
    
    # Install Python dependencies
    if not run_command("pip install -r requirements.txt", cwd=backend_dir):
        print("Failed to install Python dependencies")
        return False
    
    print("✓ Flask backend dependencies installed")
    return True

def setup_frontend():
    """Set up the React frontend."""
    print("Setting up React frontend...")
    frontend_dir = "p2p-blockchain frontend"
    
    # Install Node.js dependencies
    if not run_command("npm install", cwd=frontend_dir):
        print("Failed to install Node.js dependencies")
        return False
    
    # Build the frontend
    if not run_command("npm run build", cwd=frontend_dir):
        print("Failed to build frontend")
        return False
    
    print("✓ React frontend built successfully")
    return True

def main():
    """Main setup function."""
    print("🚀 Setting up Blockchain Integration...")
    
    # Check if we're in the right directory
    if not os.path.exists("p2p-blockchain backend") or not os.path.exists("p2p-blockchain frontend"):
        print("Error: Please run this script from the blockchain directory")
        sys.exit(1)
    
    # Setup backend
    if not setup_backend():
        print("❌ Backend setup failed")
        sys.exit(1)
    
    # Setup frontend
    if not setup_frontend():
        print("❌ Frontend setup failed")
        sys.exit(1)
    
    print("\n✅ Setup complete!")
    print("\nTo run the application:")
    print("1. Start the Flask backend:")
    print("   cd 'p2p-blockchain backend'")
    print("   python app.py")
    print("\n2. The React frontend will be served automatically at http://localhost:5000")
    print("\nThe application integrates:")
    print("- Flask backend with blockchain functionality")
    print("- React frontend with modern UI")
    print("- Real-time API communication")
    print("- CORS support for cross-origin requests")

if __name__ == "__main__":
    main()