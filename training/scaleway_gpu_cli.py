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
  %(prog)s start                           Start the GPU instance and wait until running
  %(prog)s stop                            Stop the GPU instance (async)
  %(prog)s stop-and-wait                   Stop and wait until stopped
  %(prog)s run "ls -la"                    Run a single command on the instance
  %(prog)s ssh                             Open interactive SSH session
  %(prog)s exec "cd repo" "git pull" "make"   Run multiple commands in single SSH session
        """
    )
    
    parser.add_argument(
        "command",
        choices=["start", "stop", "start-and-wait", "stop-and-wait", "run", "ssh", "exec"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "cmd",
        nargs="?",
        help="Remote command to execute (required for 'run' command) or commands for 'exec'"
    )
    
    parser.add_argument(
        "extra_cmds",
        nargs="*",
        help="Additional commands for 'exec' command"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Capture output and show after completion (default: stream in real-time)"
    )
    
    args = parser.parse_args()
    
    # Validate run command requires cmd argument
    if args.command == "run" and not args.cmd:
        parser.error("'run' command requires a command argument")
    
    # Validate exec command requires at least one cmd argument
    if args.command == "exec" and not args.cmd:
        parser.error("'exec' command requires at least one command argument")
    
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
        
        elif args.command == "run":
            interactive = not args.no_interactive
            if interactive:
                print(f"Running command: {args.cmd} (streaming output)")
            else:
                print(f"Running command: {args.cmd}")
            
            result = gpu.run_command(args.cmd, interactive=interactive)
            
            # If not interactive, print captured output
            if not interactive:
                if result.get("stdout"):
                    print(result["stdout"], end="")
                if result.get("stderr"):
                    print(result["stderr"], end="", file=sys.stderr)
            
            if not result.get("success"):
                sys.exit(result.get("returncode", 1))
        
        elif args.command == "ssh":
            print("Opening interactive SSH session...")
            returncode = gpu.open_ssh_session()
            sys.exit(returncode)
        
        elif args.command == "exec":
            # Collect all commands
            commands = [args.cmd] + (args.extra_cmds if args.extra_cmds else [])
            print(f"Executing {len(commands)} command(s) in a single SSH session...")
            for i, cmd in enumerate(commands, 1):
                print(f"  [{i}] {cmd}")
            
            interactive = not args.no_interactive
            result = gpu.run_commands(commands, interactive=interactive)
            
            # If not interactive, print captured output
            if not interactive:
                if result.get("stdout"):
                    print(result["stdout"], end="")
                if result.get("stderr"):
                    print(result["stderr"], end="", file=sys.stderr)
            
            if not result.get("success"):
                sys.exit(result.get("returncode", 1))
        
        if args.command not in ("run", "ssh", "exec"):
            print(f"Final status: {gpu.status()}")
        
    except Exception as e:
        print(f"Error executing command: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
