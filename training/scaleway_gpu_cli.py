#!/usr/bin/env python3
"""CLI for managing Scaleway GPU instances.

This script provides a command-line interface to start and stop Scaleway GPU
instances with optional wait functionality. Configuration is loaded from a .env
file at the project root.

Usage:
    python scaleway_gpu_cli.py start
    python scaleway_gpu_cli.py stop
    python scaleway_gpu_cli.py stop-and-wait

Required environment variables in .env:
    SCW_SECRET_KEY - Scaleway API secret key
    SCW_SERVER_ID  - Scaleway server/instance ID
    SCW_ZONE       - Scaleway zone (optional, default: fr-par-1)
"""

import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

from scaleway_gpu import ScalewayGPU


def load_env() -> None:
    """Load environment variables from .env file at project root."""
    root_dir = Path(__file__).parent.parent
    env_path = root_dir / ".env"
    
    if not env_path.exists():
        print(f"Error: .env file not found at {env_path}", file=sys.stderr)
        print("Please create a .env file with SCW_SECRET_KEY and SCW_SERVER_ID", file=sys.stderr)
        sys.exit(1)
    
    load_dotenv(dotenv_path=env_path)
    print(f"Loaded environment from {env_path}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Manage Scaleway GPU instances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s start              Start the GPU instance and wait until is is running
  %(prog)s stop               Stop the GPU instance (async)
  %(prog)s stop-and-wait      Stop and wait until stopped
        """
    )
    
    parser.add_argument(
        "command",
        choices=["start", "stop", "start-and-wait", "stop-and-wait"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_env()
    
    # Create GPU client
    try:
        gpu = ScalewayGPU(verbose=args.verbose)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Please ensure SCW_SECRET_KEY and SCW_SERVER_ID are set in .env", file=sys.stderr)
        sys.exit(1)
    
    # Execute command
    try:
        print(f"Current status: {gpu.status()}")
                
        if args.command == "stop":
            print("Stopping GPU instance...")
            result = gpu.stop()
            if result.get("skipped"):
                print(f"✓ {result.get('message')}")
            else:
                print("✓ Stop command sent successfully")
                print(f"Instance state: {gpu.status()}")
        
        elif args.command == "start":
            print("Starting GPU instance and waiting for 'running' state...")
            success = gpu.start_and_wait()
            if success:
                print("✓ Instance is now running")
            else:
                print("✗ Timeout: Instance did not reach 'running' state", file=sys.stderr)
                sys.exit(1)
        
        elif args.command == "stop-and-wait":
            print("Stopping GPU instance and waiting for 'stopped' state...")
            success = gpu.stop_and_wait()
            if success:
                print("✓ Instance is now stopped")
            else:
                print("✗ Timeout: Instance did not reach 'stopped' state", file=sys.stderr)
                sys.exit(1)
        
        print(f"Final status: {gpu.status()}")
        
    except Exception as e:
        print(f"Error executing command: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
