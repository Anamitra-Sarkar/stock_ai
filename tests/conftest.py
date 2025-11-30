"""
Pytest configuration and fixtures for the stock_ai test suite.
"""

import pytest


def pytest_addoption(parser):
    """Add command line options for performance tests"""
    parser.addoption(
        "--load-test",
        action="store_true",
        default=False,
        help="Run load testing scenarios",
    )
