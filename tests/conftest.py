"""Shared test fixtures for agent-mesh-router.

Fixtures defined here are available to all tests in the suite without
needing an explicit import. Add project-wide fixtures here; keep
domain-specific fixtures close to the tests that use them.
"""
from __future__ import annotations

import pytest


@pytest.fixture()
def package_name() -> str:
    """Return the importable package name for assertions."""
    return "agent_mesh_router"


@pytest.fixture()
def expected_version() -> str:
    """Return the current expected version string.

    Update this fixture when cutting a release so that the version
    test immediately catches stale ``__version__`` values.
    """
    return "0.1.0"
