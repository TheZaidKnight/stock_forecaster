#!/usr/bin/env python3
"""
Stock Forecaster Application Launcher
=====================================

This script provides an easy way to run the Stock Forecaster application
with different configurations.

Usage:
    python run.py                    # Run in development mode
    python run.py --production       # Run in production mode
    python run.py --test             # Run tests
    python run.py --help             # Show help
"""

import argparse
import sys
import os
import subprocess

def run_application(mode='development'):
    """Run the Flask application in the specified mode."""
    os.environ['FLASK_ENV'] = mode
    
    if mode == 'production':
        print("🚀 Starting Stock Forecaster in PRODUCTION mode...")
        print("⚠️  Make sure to set proper environment variables!")
        os.system("cd backend && py -3.12 app.py")
    else:
        print("🔧 Starting Stock Forecaster in DEVELOPMENT mode...")
        os.system("cd backend && py -3.12 app.py")

def run_tests():
    """Run the test suite."""
    print("🧪 Running Stock Forecaster test suite...")
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            'backend/tests/', '-v', '--tb=short'
        ], cwd=os.getcwd())
        return result.returncode == 0
    except FileNotFoundError:
        print("❌ pytest not found. Install it with: pip install pytest")
        return False

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("🔍 Checking dependencies...")
    try:
        import yfinance, requests, pandas, plotly, flask, tensorflow, statsmodels
        print("✅ All required dependencies are installed!")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("📦 Install dependencies with: pip install -r requirements.txt")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Stock Forecaster Application Launcher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                    # Run in development mode
  python run.py --production       # Run in production mode  
  python run.py --test             # Run test suite
  python run.py --check            # Check dependencies only
        """
    )
    
    parser.add_argument('--production', action='store_true',
                       help='Run in production mode')
    parser.add_argument('--test', action='store_true',
                       help='Run the test suite')
    parser.add_argument('--check', action='store_true',
                       help='Check dependencies only')
    
    args = parser.parse_args()
    
    print("📈 Stock Forecaster Application")
    print("=" * 40)
    
    if args.check:
        check_dependencies()
        return
    
    if args.test:
        if check_dependencies():
            success = run_tests()
            if success:
                print("✅ All tests passed!")
            else:
                print("❌ Some tests failed!")
                sys.exit(1)
        return
    
    if not check_dependencies():
        print("❌ Please install missing dependencies first.")
        sys.exit(1)
    
    if args.production:
        run_application('production')
    else:
        run_application('development')

if __name__ == '__main__':
    main()
