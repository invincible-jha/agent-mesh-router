"""Routing subsystem for agent-mesh-router.

Exports the routing strategy ABC, five concrete strategy implementations,
the RoutingTable, and the RouteResolver.

Example
-------
::

    from agent_mesh_router.routing import (
        RoutingTable,
        RouteResolver,
        CapabilityMatchRouter,
        CostAwareRouter,
        RoundRobinRouter,
        LeastLoadedRouter,
        CompositeRouter,
    )

    table = RoutingTable()
    table.register_agent(
        agent_id="summarizer",
        capabilities=["summarize", "text"],
        cost_per_token=0.000002,
        current_load=0.1,
    )
    router = CapabilityMatchRouter()
    resolver = RouteResolver(table=table, strategy=router)
"""
from __future__ import annotations

from agent_mesh_router.routing.resolver import RouteResolver
from agent_mesh_router.routing.strategy import (
    CapabilityMatchRouter,
    CompositeRouter,
    CostAwareRouter,
    LeastLoadedRouter,
    RoundRobinRouter,
    RoutingStrategy,
)
from agent_mesh_router.routing.table import AgentRecord, RoutingTable

__all__ = [
    "RoutingStrategy",
    "CapabilityMatchRouter",
    "CostAwareRouter",
    "RoundRobinRouter",
    "LeastLoadedRouter",
    "CompositeRouter",
    "RoutingTable",
    "AgentRecord",
    "RouteResolver",
]
