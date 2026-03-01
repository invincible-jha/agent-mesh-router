#!/usr/bin/env python3
"""Example: Routing Strategies

Demonstrates the built-in routing strategies: RoundRobin, LeastLoaded,
CapabilityMatch, and Composite routing.

Usage:
    python examples/02_routing_strategies.py

Requirements:
    pip install agent-mesh-router
"""
from __future__ import annotations

import agent_mesh_router
from agent_mesh_router import (
    AgentRecord,
    CapabilityMatchRouter,
    CompositeRouter,
    LeastLoadedRouter,
    MessageEnvelope,
    MessageType,
    Priority,
    RoundRobinRouter,
    RoutingTable,
)


def build_table() -> RoutingTable:
    """Build a routing table with agents at different loads."""
    table = RoutingTable()
    agents = [
        ("agent-a", ["nlp", "summarise"], 0.2),
        ("agent-b", ["nlp", "translate"], 0.8),
        ("agent-c", ["code", "review"], 0.1),
        ("agent-d", ["nlp", "summarise", "review"], 0.4),
    ]
    for agent_id, caps, load in agents:
        table.register(AgentRecord(
            agent_id=agent_id,
            capabilities=caps,
            load=load,
        ))
    return table


def make_envelope(sender: str, task: str) -> MessageEnvelope:
    return MessageEnvelope(
        sender_id=sender,
        recipient_id="",
        message_type=MessageType.TASK,
        priority=Priority.NORMAL,
        payload={"task": task},
    )


def main() -> None:
    print(f"agent-mesh-router version: {agent_mesh_router.__version__}")

    table = build_table()
    print(f"Agents registered: {table.count()}")

    # Strategy 1: Round-robin
    rr = RoundRobinRouter(table=table)
    envelope = make_envelope("orchestrator", "summarise doc")
    result1 = rr.select(envelope)
    result2 = rr.select(envelope)
    print(f"\nRound-robin: {result1.agent_id} -> {result2.agent_id}")

    # Strategy 2: Least loaded
    ll = LeastLoadedRouter(table=table)
    result = ll.select(envelope)
    print(f"Least loaded: {result.agent_id} (load={result.load:.1f})")

    # Strategy 3: Capability match
    cap_envelope = make_envelope("orchestrator", "nlp task")
    cap_envelope = cap_envelope.model_copy(
        update={"required_capabilities": ["nlp", "summarise"]}
    )
    cm = CapabilityMatchRouter(table=table)
    result = cm.select(cap_envelope)
    print(f"Capability match: {result.agent_id} (caps={result.capabilities})")

    # Strategy 4: Composite (capability first, then least loaded)
    composite = CompositeRouter(strategies=[cm, ll])
    result = composite.select(cap_envelope)
    print(f"Composite: {result.agent_id}")


if __name__ == "__main__":
    main()
