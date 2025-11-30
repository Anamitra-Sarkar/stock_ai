#!/usr/bin/env python3
"""
Quick deployment test script for Stock AI Platform
Tests core functionality without external dependencies
"""
import os
import sys
import requests
import time
import subprocess
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_deployment_startup():
    """Test app starts successfully with deployment flags"""
    print("🧪 Testing deployment startup...")

    # Set deployment environment variables
    env = os.environ.copy()
    env.update(
        {
            "MINIMAL_STARTUP": "true",
            "SKIP_ML_TRAINING": "true",
            "SECRET_KEY": "test-secret-key",
        }
    )

    # Start app in background
    process = subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=project_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait for startup
    time.sleep(15)

    try:
        # Check if process is still running
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            print(f"❌ App failed to start")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return False

        # Test health endpoint
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health endpoint working")
            health_data = response.json()
            print(f"   Status: {health_data.get('status')}")
            print(f"   Services: {health_data.get('services', {})}")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False

        # Test API status endpoint
        response = requests.get("http://localhost:5000/api/status", timeout=5)
        if response.status_code == 200:
            print("✅ API status endpoint working")
            status_data = response.json()
            print(f"   Cache: {status_data.get('cache_stats', {}).get('backend')}")
            print(
                f"   API Key: {'Present' if status_data.get('api_key_configured') else 'Missing'}"
            )
        else:
            print(f"❌ API status failed: {response.status_code}")

        # Test dashboard endpoint
        response = requests.get("http://localhost:5000/api/dashboard", timeout=10)
        if response.status_code == 200:
            print("✅ Dashboard endpoint working")
            dashboard_data = response.json()
            print(f"   Total stocks: {len(dashboard_data.get('trending_stocks', []))}")
        else:
            print(f"❌ Dashboard failed: {response.status_code}")

        print("✅ Deployment test passed!")
        return True

    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

    finally:
        # Clean up process
        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def test_import_modules():
    """Test that all modules can be imported"""
    print("\n🧪 Testing module imports...")

    modules_to_test = [
        "config",
        "cache.redis_cache",
        "agents.prediction_agent",
        "agents.sentiment_agent",
        "agents.advisor_agent",
        "agents.alert_agent",
        "streaming.websocket_manager",
        "database.connection",
        "portfolio.optimizer",
        "risk.risk_manager",
    ]

    failed_imports = []

    for module in modules_to_test:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except Exception as e:
            print(f"   ❌ {module}: {e}")
            failed_imports.append(module)

    if failed_imports:
        print(f"\n❌ Failed to import: {failed_imports}")
        return False
    else:
        print("✅ All modules imported successfully!")
        return True


def test_configuration():
    """Test configuration loading"""
    print("\n🧪 Testing configuration...")

    try:
        from config import config

        print(f"   Host: {config.host}")
        print(f"   Port: {config.port}")
        print(f"   Debug: {config.debug}")
        print(f"   Redis host: {config.redis.host}")
        print(f"   DB host: {config.database.host}")
        print("✅ Configuration loaded successfully!")
        return True

    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        return False


def main():
    """Run all deployment tests"""
    print("🚀 Stock AI Platform Deployment Tests")
    print("=" * 50)

    # Change to project directory
    os.chdir(project_root)

    tests = [test_configuration, test_import_modules, test_deployment_startup]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)

    print("\n" + "=" * 50)
    print("🏁 Test Results:")
    print(f"   Passed: {sum(results)}/{len(results)}")

    if all(results):
        print("✅ All deployment tests passed! App is ready for deployment.")
        return 0
    else:
        print("❌ Some tests failed. Check issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
