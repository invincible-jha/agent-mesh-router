"""Deadlock detection for agent dependency graphs.

Detects cycles in agent wait-for relationships using depth-first search.

Classes
-------
DeadlockDetector
    Build and analyse a directed wait-for graph.  Call
    :meth:`~DeadlockDetector.detect` to find all cycles.
DeadlockResult
    Immutable result of a detection run — carries cycle paths and metadata.
"""
from __future__ import annotations

from agent_mesh_router.deadlock.detector import (
    DeadlockDetector,
    DeadlockResult,
    WaitEdge,
)

__all__ = [
    "DeadlockDetector",
    "DeadlockResult",
    "WaitEdge",
]
