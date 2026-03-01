#!/usr/bin/env python3
"""Example: Fleet Registry and Health Monitoring

Demonstrates agent node registration, status tracking, health
monitoring, and load-balanced agent selection.

Usage:
    python examples/04_fleet_registry.py

Requirements:
    pip install agent-mesh-router
"""
from __future__ import annotations

import agent_mesh_router
from agent_mesh_router import (
    AgentNode,
    AgentStatus,
    FleetRegistry,
    HealthMonitor,
    LoadBalancer,
    Strategy,
)


def main() -> None:
    print(f"agent-mesh-router version: {agent_mesh_router.__version__}")

    # Step 1: Build fleet registry
    registry = FleetRegistry()
    nodes = [
        AgentNode(
            node_id="node-nlp-1",
            capabilities=["nlp", "summarise"],
            endpoint="http://nlp-1:8080",
            max_concurrent=10,
        ),
        AgentNode(
            node_id="node-nlp-2",
            capabilities=["nlp", "translate"],
            endpoint="http://nlp-2:8080",
            max_concurrent=8,
        ),
        AgentNode(
            node_id="node-code-1",
            capabilities=["code", "review"],
            endpoint="http://code-1:8080",
            max_concurrent=5,
        ),
    ]
    for node in nodes:
        registry.register(node)
    print(f"Fleet size: {registry.count()} nodes")

    # Step 2: Update node statuses
    registry.set_status("node-nlp-2", AgentStatus.BUSY)
    available = registry.list_by_status(AgentStatus.AVAILABLE)
    print(f"Available nodes: {[n.node_id for n in available]}")

    # Step 3: Health monitoring
    monitor = HealthMonitor(registry=registry)
    monitor.record_heartbeat("node-nlp-1")
    monitor.record_heartbeat("node-code-1")
    healthy = monitor.get_healthy_nodes()
    print(f"Healthy nodes: {[n.node_id for n in healthy]}")

    # Step 4: Load balancing across available nodes
    lb = LoadBalancer(registry=registry, strategy=Strategy.ROUND_ROBIN)
    for capability in ["nlp", "nlp", "code"]:
        try:
            selected = lb.select(required_capability=capability)
            print(f"Selected for '{capability}': {selected.node_id}")
        except Exception as error:
            print(f"  No node for '{capability}': {error}")

    # Step 5: Deregister a node
    registry.deregister("node-code-1")
    print(f"\nAfter deregister: {registry.count()} nodes")


if __name__ == "__main__":
    main()
