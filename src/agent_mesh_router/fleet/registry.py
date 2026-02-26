"""FleetRegistry — live registry of agent nodes in the mesh.

Tracks which agents are running, what capabilities they expose, how busy
they are (load_score), and when they last sent a heartbeat.  Agents that
miss heartbeats for longer than a configurable staleness threshold are
automatically pruned (or marked UNHEALTHY by the HealthMonitor).

Design
------
- ``AgentNode`` is a plain dataclass — serializable and cheaply copied.
- ``FleetRegistry`` is thread-safe via a ``threading.RLock``.
- ``get_available(capability)`` returns only nodes that are HEALTHY and
  expose the requested capability.
- ``prune_stale(timeout)`` removes nodes whose last heartbeat is older
  than ``timeout`` seconds.

Example
-------
::

    from agent_mesh_router.fleet.registry import FleetRegistry, AgentNode

    registry = FleetRegistry()
    registry.register(AgentNode(
        agent_id="worker-1",
        capabilities={"summarize", "translate"},
        status="HEALTHY",
        load_score=0.2,
    ))

    nodes = registry.get_available("summarize")
    print(len(nodes))  # 1
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    """Lifecycle status of an agent node."""

    HEALTHY = "HEALTHY"
    UNHEALTHY = "UNHEALTHY"
    DEGRADED = "DEGRADED"
    STARTING = "STARTING"
    STOPPING = "STOPPING"


@dataclass
class AgentNode:
    """A single agent node in the fleet.

    Attributes
    ----------
    agent_id:
        Unique identifier for this agent.
    capabilities:
        Set of capability strings the agent exposes.
    status:
        Current lifecycle status.  Only HEALTHY nodes are returned by
        ``get_available``.
    load_score:
        Normalized load in [0.0, 1.0]; 0.0 = idle, 1.0 = fully loaded.
    last_heartbeat:
        Unix epoch seconds of the most recent heartbeat received.
        Defaults to ``time.time()`` at construction.
    metadata:
        Free-form key→value metadata (region, version, model, etc.).
    """

    agent_id: str
    capabilities: set[str] = field(default_factory=set)
    status: AgentStatus = AgentStatus.HEALTHY
    load_score: float = 0.0
    last_heartbeat: float = field(default_factory=time.time)
    metadata: dict[str, str] = field(default_factory=dict)

    def is_healthy(self) -> bool:
        """Return True when status is HEALTHY."""
        return self.status == AgentStatus.HEALTHY

    def has_capability(self, capability: str) -> bool:
        """Return True if this node exposes ``capability``."""
        return capability in self.capabilities


class AgentNodeNotFoundError(KeyError):
    """Raised when an agent_id is not present in the FleetRegistry."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        super().__init__(f"AgentNode {agent_id!r} is not registered.")


class FleetRegistry:
    """Thread-safe registry of active agent nodes.

    Parameters
    ----------
    default_status:
        Status assigned to newly registered nodes.
    """

    def __init__(
        self,
        *,
        default_status: AgentStatus = AgentStatus.HEALTHY,
    ) -> None:
        self._default_status = default_status
        self._nodes: dict[str, AgentNode] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, node: AgentNode) -> None:
        """Add or replace an agent node in the registry.

        If a node with the same ``agent_id`` already exists, it is
        overwritten with the new node object.

        Parameters
        ----------
        node:
            The AgentNode to register.  The node is stored by reference;
            external mutations to the object will be reflected.

        Raises
        ------
        ValueError
            If ``node.load_score`` is outside [0.0, 1.0].
        """
        if not (0.0 <= node.load_score <= 1.0):
            raise ValueError(
                f"AgentNode.load_score must be in [0.0, 1.0], "
                f"got {node.load_score}."
            )
        with self._lock:
            self._nodes[node.agent_id] = node
        logger.debug("Fleet: registered agent %r.", node.agent_id)

    def deregister(self, agent_id: str) -> None:
        """Remove an agent node from the registry.

        Parameters
        ----------
        agent_id:
            ID of the node to remove.

        Raises
        ------
        AgentNodeNotFoundError
            If the agent is not registered.
        """
        with self._lock:
            if agent_id not in self._nodes:
                raise AgentNodeNotFoundError(agent_id)
            del self._nodes[agent_id]
        logger.debug("Fleet: deregistered agent %r.", agent_id)

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    def heartbeat(
        self,
        agent_id: str,
        *,
        load_score: float | None = None,
        status: AgentStatus | None = None,
    ) -> None:
        """Record a heartbeat for an agent.

        Updates ``last_heartbeat`` to the current time and optionally
        updates ``load_score`` and ``status``.

        Parameters
        ----------
        agent_id:
            The agent sending the heartbeat.
        load_score:
            New load value; if None the existing value is preserved.
        status:
            New status; if None the existing status is preserved.

        Raises
        ------
        AgentNodeNotFoundError
            If the agent is not registered.
        ValueError
            If ``load_score`` is outside [0.0, 1.0].
        """
        if load_score is not None and not (0.0 <= load_score <= 1.0):
            raise ValueError(
                f"load_score must be in [0.0, 1.0], got {load_score}."
            )
        with self._lock:
            node = self._get_or_raise(agent_id)
            node.last_heartbeat = time.time()
            if load_score is not None:
                node.load_score = load_score
            if status is not None:
                node.status = status
        logger.debug("Fleet: heartbeat from %r.", agent_id)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, agent_id: str) -> AgentNode:
        """Return the node registered under ``agent_id``.

        Raises
        ------
        AgentNodeNotFoundError
        """
        with self._lock:
            return self._get_or_raise(agent_id)

    def get_available(self, capability: str) -> list[AgentNode]:
        """Return all HEALTHY nodes that expose ``capability``.

        Parameters
        ----------
        capability:
            Capability string to filter on.

        Returns
        -------
        list[AgentNode]
            Healthy nodes with the capability, unordered.  Empty list if none.
        """
        with self._lock:
            return [
                node
                for node in self._nodes.values()
                if node.is_healthy() and node.has_capability(capability)
            ]

    def list_nodes(
        self,
        *,
        healthy_only: bool = False,
    ) -> list[AgentNode]:
        """Return a snapshot of all registered nodes.

        Parameters
        ----------
        healthy_only:
            When True, only HEALTHY nodes are returned.

        Returns
        -------
        list[AgentNode]
            Copy of the node list; mutations to the list do not affect the registry.
        """
        with self._lock:
            nodes = list(self._nodes.values())

        if healthy_only:
            return [n for n in nodes if n.is_healthy()]
        return nodes

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def prune_stale(self, timeout: float) -> list[str]:
        """Remove nodes whose last heartbeat is older than ``timeout`` seconds.

        Parameters
        ----------
        timeout:
            Staleness threshold in seconds.  Nodes with
            ``time.time() - last_heartbeat > timeout`` are removed.

        Returns
        -------
        list[str]
            Agent IDs of the nodes that were pruned.
        """
        now = time.time()
        stale_ids: list[str] = []

        with self._lock:
            for agent_id, node in list(self._nodes.items()):
                if (now - node.last_heartbeat) > timeout:
                    stale_ids.append(agent_id)
            for agent_id in stale_ids:
                del self._nodes[agent_id]

        if stale_ids:
            logger.info(
                "Fleet: pruned %d stale agent(s): %r", len(stale_ids), stale_ids
            )
        return stale_ids

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._nodes)

    def __contains__(self, agent_id: object) -> bool:
        with self._lock:
            return agent_id in self._nodes

    def __repr__(self) -> str:
        with self._lock:
            return f"FleetRegistry(agents={list(self._nodes.keys())!r})"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_raise(self, agent_id: str) -> AgentNode:
        try:
            return self._nodes[agent_id]
        except KeyError:
            raise AgentNodeNotFoundError(agent_id) from None
