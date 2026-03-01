#!/usr/bin/env python3
"""Example: Quickstart — agent-mesh-router

Minimal working example: register agents, route a message, and run
a sequential workflow.

Usage:
    python examples/01_quickstart.py

Requirements:
    pip install agent-mesh-router
"""
from __future__ import annotations

import agent_mesh_router
from agent_mesh_router import (
    Router,
    MessageEnvelope,
    MessageType,
    Priority,
    RoutingTable,
    AgentRecord,
    SequentialWorkflow,
    WorkflowStep,
)


def make_step(name: str, task: str) -> WorkflowStep:
    """Create a workflow step with a simple callable."""
    def handler(payload: dict[str, object]) -> dict[str, object]:
        return {"step": name, "result": f"[{name}] processed: {task}"}
    return WorkflowStep(name=name, handler=handler)


def main() -> None:
    print(f"agent-mesh-router version: {agent_mesh_router.__version__}")

    # Step 1: Build a routing table with two agents
    table = RoutingTable()
    for agent_id, capabilities in [
        ("summariser-agent", ["nlp", "summarise"]),
        ("reviewer-agent", ["nlp", "review"]),
    ]:
        table.register(AgentRecord(
            agent_id=agent_id,
            capabilities=capabilities,
            load=0.0,
        ))
    print(f"Routing table: {table.count()} agents registered")

    # Step 2: Route a message using the convenience Router
    router = Router(table=table)
    envelope = MessageEnvelope(
        sender_id="orchestrator",
        recipient_id="summariser-agent",
        message_type=MessageType.TASK,
        priority=Priority.HIGH,
        payload={"text": "Summarise the quarterly report."},
    )
    resolved = router.route(envelope)
    print(f"Routed to: {resolved.recipient_id} (priority={resolved.priority.value})")

    # Step 3: Run a sequential workflow
    workflow = SequentialWorkflow(steps=[
        make_step("fetch", "retrieve document"),
        make_step("summarise", "condense to 3 bullets"),
        make_step("review", "check quality"),
    ])
    result = workflow.run(payload={})
    print(f"\nWorkflow status: {result.status.value}")
    for step_result in result.step_results:
        print(f"  [{step_result.step_name}] {step_result.output}")


if __name__ == "__main__":
    main()
