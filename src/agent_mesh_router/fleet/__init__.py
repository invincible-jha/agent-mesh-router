"""Fleet management — agent node registry, health monitoring, and load balancing.

This package provides the runtime view of the agent fleet:

registry
    ``FleetRegistry`` and ``AgentNode`` — register/deregister agents,
    record heartbeats, and query available nodes by capability.

health
    ``HealthMonitor`` — background task that marks agents UNHEALTHY when
    they stop sending heartbeats.

load_balancer
    ``LoadBalancer`` with ``Strategy`` enum — select agents using
    round-robin, least-loaded, random, or capability-weighted strategies.

Example
-------
::

    from agent_mesh_router.fleet import (
        AgentNode,
        AgentStatus,
        FleetRegistry,
        HealthMonitor,
        LoadBalancer,
        Strategy,
    )
"""
from __future__ import annotations

from agent_mesh_router.fleet.health import HealthMonitor
from agent_mesh_router.fleet.load_balancer import LoadBalancer, Strategy
from agent_mesh_router.fleet.registry import (
    AgentNode,
    AgentNodeNotFoundError,
    AgentStatus,
    FleetRegistry,
)

__all__ = [
    "AgentNode",
    "AgentNodeNotFoundError",
    "AgentStatus",
    "FleetRegistry",
    "HealthMonitor",
    "LoadBalancer",
    "Strategy",
]
