"""Convenience API for agent-mesh-router — 3-line quickstart.

Example
-------
::

    from agent_mesh_router import Router
    router = Router()
    result = router.route({"type": "task", "content": "Process this"})

"""
from __future__ import annotations

from typing import Any


class Router:
    """Zero-config mesh router for the 80% use case.

    Provides a simple route() interface backed by a RoutingTable and
    RoundRobinRouter with no external dependencies required.

    Example
    -------
    ::

        from agent_mesh_router import Router
        router = Router()
        # Register an agent endpoint
        router.register("worker-1", capabilities=["nlp", "search"])
        # Route a task
        result = router.route({"type": "search", "query": "Python AI"})
    """

    def __init__(self) -> None:
        from agent_mesh_router.routing.table import RoutingTable
        from agent_mesh_router.routing.strategy import RoundRobinRouter

        self._table = RoutingTable()
        self._strategy = RoundRobinRouter()

    def register(
        self,
        agent_id: str,
        capabilities: list[str] | None = None,
    ) -> None:
        """Register an agent endpoint with the routing table.

        Parameters
        ----------
        agent_id:
            Unique agent identifier.
        capabilities:
            List of capability strings for capability-based routing.
        """
        self._table.register_agent(
            agent_id=agent_id,
            capabilities=set(capabilities or []),
        )

    def route(self, task: dict[str, Any]) -> dict[str, Any]:
        """Route a task to the best available agent.

        Parameters
        ----------
        task:
            Task descriptor dict. At minimum includes ``type`` and
            optionally ``capabilities`` to request.

        Returns
        -------
        dict[str, Any]
            Routing decision with ``agent_id`` key, or an error dict
            if no agents are available.

        Example
        -------
        ::

            router = Router()
            router.register("worker-1", capabilities=["nlp"])
            result = router.route({"type": "nlp", "content": "Hello"})
            print(result["agent_id"])
        """
        from agent_mesh_router.messages.envelope import MessageEnvelope
        from agent_mesh_router.messages.types import MessageType, Priority

        agents = self._table.list_agents()
        if not agents:
            return {"error": "no_agents_available", "task": task}

        # Build a minimal envelope to pass to the routing strategy
        envelope = MessageEnvelope(
            sender="router-quickstart",
            receiver="*",
            message_type=MessageType.TASK,
            payload=task,
            priority=Priority.NORMAL,
        )

        selected = self._strategy.select(self._table, envelope)
        if selected is None:
            return {"error": "no_matching_agent", "task": task}

        return {
            "agent_id": selected.agent_id,
            "task": task,
        }

    @property
    def table(self) -> Any:
        """The underlying RoutingTable instance."""
        return self._table

    def __repr__(self) -> str:
        agent_count = len(self._table.list_agents())
        return f"Router(agents={agent_count}, strategy=RoundRobin)"
