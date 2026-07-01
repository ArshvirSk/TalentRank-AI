"""
conftest.py — pytest configuration for Person B test suite.

Registers custom marks and prints diagnostic info on test failure.
Requirements: 13.6
"""
import pytest


def pytest_configure(config):
    """Register custom marks so pytest doesn't warn about unknown marks."""
    config.addinivalue_line(
        "markers",
        "fast: tests that run without any model download (sentinel/text construction only)",
    )
    config.addinivalue_line(
        "markers",
        "slow: tests that require the embedding model to be downloaded",
    )


def pytest_runtest_logreport(report):
    """On test failure, print captured stdout so debug prints are visible.

    Tests use direct print() calls before assertions to emit candidate text
    or score values. This hook ensures those prints surface in the failure
    output even when pytest's default capture would hide them.

    Requirements: 13.6
    """
    if report.when == "call" and report.failed:
        # report.capstdout contains everything printed during the test
        if hasattr(report, "capstdout") and report.capstdout:
            print(f"\n--- captured stdout (diagnostic) ---\n{report.capstdout}")
        if hasattr(report, "longreprtext"):
            pass  # already shown by pytest; no duplication needed
