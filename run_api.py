#!/usr/bin/env python3
"""
Script to run the Sarcasm Detection API server.
"""

import os
import sys
import uvicorn
from pathlib import Path

def main():
    """Run the FastAPI server."""
    
    # Add src to path
    src_path = Path(__file__).parent / 'src'
    sys.path.append(str(src_path))
    
    # Change to src/api directory
    api_path = src_path / 'api'
    os.chdir(api_path)
    
    print("🚀 Starting Sarcasm Detection API...")
    print(f"📁 Working directory: {api_path}")
    print("🌐 API will be available at: http://localhost:8000")
    print("📚 API documentation: http://localhost:8000/docs")
    print("🔍 Health check: http://localhost:8000/health")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        # Run the FastAPI server
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have installed dependencies: pip install -r requirements.txt")
        print("2. Check if the model file exists: models/best_model.pkl")
        print("3. Verify the API code is correct")

if __name__ == "__main__":
    main() 