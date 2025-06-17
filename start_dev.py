#!/usr/bin/env python3
"""
Development Startup Script for Traffic Monitor
Provides easy way to start the system with different configurations
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_venv():
    """Check if we're in a virtual environment"""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def install_requirements():
    """Install required packages"""
    print("📦 Installing/updating requirements...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def start_development_server(mode='dev'):
    """Start the development server"""
    print(f"🚀 Starting traffic monitor in {mode} mode...")
    
    if mode == 'dev':
        subprocess.run([sys.executable, 'dev_monitor.py'])
    elif mode == 'original':
        subprocess.run([sys.executable, 'o4_dg_monitor.py'])
    else:
        print(f"❌ Unknown mode: {mode}")

def main():
    parser = argparse.ArgumentParser(description='Traffic Monitor Development Launcher')
    parser.add_argument('--mode', choices=['dev', 'original'], default='dev',
                       help='Run mode: dev (development with webcam) or original (hardware)')
    parser.add_argument('--install', action='store_true',
                       help='Install/update requirements before starting')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check environment, don\'t start server')
    
    args = parser.parse_args()
    
    print("🔧 Traffic Monitor Development Launcher")
    print("=" * 40)
    
    # Check virtual environment
    if not check_venv():
        print("⚠️  Not running in virtual environment!")
        print("💡 Activate with: source venv/bin/activate")
        sys.exit(1)
    
    print(f"✅ Virtual environment active: {sys.prefix}")
    
    # Install requirements if requested
    if args.install:
        if not install_requirements():
            sys.exit(1)
    
    # Check if required files exist
    required_files = {
        'dev': ['dev_monitor.py', 'config_dev.py', 'templates/dashboard_final.html'],
        'original': ['o4_dg_monitor.py', 'o4_hailo_detection_module.py']
    }
    
    missing_files = []
    for file in required_files[args.mode]:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files for {args.mode} mode:")
        for file in missing_files:
            print(f"   - {file}")
        sys.exit(1)
    
    if args.check_only:
        print("✅ Environment check complete - all good!")
        return
    
    # Start the server
    try:
        start_development_server(args.mode)
    except KeyboardInterrupt:
        print("\n⚠️  Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == '__main__':
    main()