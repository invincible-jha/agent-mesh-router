"""Agent resource quota enforcement.

Provides per-agent resource quota definitions and enforcement:

ResourceQuota
    Dataclass declaring per-dimension limits (tokens, memory, CPU, calls).
QuotaEnforcer
    Thread-safe enforcer that tracks usage and blocks over-quota requests.
QuotaViolation
    Exception raised when a quota would be exceeded.

Classes
-------
ResourceQuota
    Defines limits for one or more resource dimensions.
QuotaUsage
    Point-in-time snapshot of consumption per dimension.
QuotaEnforcer
    Registers quotas and checks/records resource usage.
QuotaViolation
    Exception type for quota breaches.
"""
from __future__ import annotations

from agent_mesh_router.quotas.resource_quota import (
    QuotaEnforcer,
    QuotaUsage,
    QuotaViolation,
    ResourceQuota,
)

__all__ = [
    "QuotaEnforcer",
    "QuotaUsage",
    "QuotaViolation",
    "ResourceQuota",
]
