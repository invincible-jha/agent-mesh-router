#!/usr/bin/env python3
"""Example: LangChain Routing Integration

Demonstrates wrapping LangChain chains as routable agent nodes
and dispatching tasks through the mesh router.

Usage:
    python examples/07_langchain_routing.py

Requirements:
    pip install agent-mesh-router
    pip install langchain   # optional — example degrades gracefully
"""
from __future__ import annotations

import agent_mesh_router
from agent_mesh_router import (
    AgentNode,
    FleetRegistry,
    LoadBalancer,
    MessageEnvelope,
    MessageType,
    Priority,
    Router,
    RoutingTable,
    AgentRecord,
    Strategy,
    TracingMiddleware,
)

try:
    from langchain.schema.runnable import RunnableLambda
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False


def build_langchain_node(node_id: str, capability: str) -> "object":
    """Wrap a callable as a LangChain runnable (stub when LC not installed)."""
    if not _LANGCHAIN_AVAILABLE:
        return None

    def process(inputs: dict[str, object]) -> dict[str, object]:
        return {"node": node_id, "output": f"[{capability}] processed input"}

    return RunnableLambda(process)


def dispatch(
    router: Router,
    table: RoutingTable,
    task: str,
    lc_nodes: dict[str, "object"],
) -> None:
    """Route a task and invoke the matching LangChain runnable."""
    envelope = MessageEnvelope(
        sender_id="orchestrator",
        recipient_id="summariser-1",
        message_type=MessageType.TASK,
        priority=Priority.NORMAL,
        payload={"task": task},
    )
    resolved = router.route(envelope)
    print(f"  Routed '{task[:40]}' -> {resolved.recipient_id}")

    if _LANGCHAIN_AVAILABLE and resolved.recipient_id in lc_nodes:
        runnable = lc_nodes[resolved.recipient_id]
        result = runnable.invoke({"task": task})  # type: ignore[union-attr]
        print(f"  LangChain result: {result}")
    else:
        print(f"  (LangChain not installed — routing only)")


def main() -> None:
    print(f"agent-mesh-router version: {agent_mesh_router.__version__}")

    if not _LANGCHAIN_AVAILABLE:
        print("LangChain not installed — demonstrating routing only.")
        print("Install with: pip install langchain")

    # Build routing table
    table = RoutingTable()
    for agent_id, caps in [
        ("summariser-1", ["nlp", "summarise"]),
        ("reviewer-1", ["nlp", "review"]),
    ]:
        table.register(AgentRecord(
            agent_id=agent_id,
            capabilities=caps,
            load=0.0,
        ))

    # Build router with tracing middleware
    router = Router(table=table, middleware=[TracingMiddleware()])

    # Build LangChain nodes (or stubs)
    lc_nodes: dict[str, object] = {
        "summariser-1": build_langchain_node("summariser-1", "summarise"),
        "reviewer-1": build_langchain_node("reviewer-1", "review"),
    }

    print(f"\nDispatching tasks to {table.count()} agents:")
    dispatch(router, table, "Summarise the annual report into bullet points.", lc_nodes)
    dispatch(router, table, "Review the draft proposal for clarity.", lc_nodes)


if __name__ == "__main__":
    main()
