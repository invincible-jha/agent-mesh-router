"""RoutingTable — maps agent IDs to capabilities, load, and cost metadata.

The RoutingTable is the source of truth for what agents are available in
the mesh and what they can do.  Routing strategies query this table to
make decisions without needing direct knowledge of individual agents.

Thread-safety: all mutations are protected by a ``threading.RLock`` so the
table can be shared between sync and async code safely.

Example
-------
::

    from agent_mesh_router.routing.table import RoutingTable

    table = RoutingTable()
    table.register_agent(
        agent_id="coder",
        capabilities=["code_generation", "python"],
        cost_per_token=0.000004,
        current_load=0.3,
        tags={"region": "us-east-1"},
    )

    record = table.get_agent("coder")
    capable = table.find_by_capability("python")
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class AgentRecord:
    """Snapshot of an agent's registration state.

    Attributes
    ----------
    agent_id:
        Unique agent identifier within the mesh.
    capabilities:
        Set of capability strings this agent can handle (e.g. ``"summarize"``).
    cost_per_token:
        Estimated USD cost per LLM token for this agent.  Used by the
        ``CostAwareRouter``.
    current_load:
        Normalized load value in the range [0.0, 1.0].  Higher means busier.
        Used by the ``LeastLoadedRouter``.
    healthy:
        Whether the agent is currently considered healthy.  Unhealthy agents
        are excluded from routing by default.
    registered_at:
        Unix epoch seconds when this record was first created.
    last_heartbeat_at:
        Unix epoch seconds of the most recent heartbeat from this agent.
    tags:
        Arbitrary key→value metadata for filtering and observability.
    metadata:
        Extended attributes not covered by the typed fields above.
    """

    agent_id: str
    capabilities: set[str] = field(default_factory=set)
    cost_per_token: float = 0.0
    current_load: float = 0.0
    healthy: bool = True
    registered_at: float = field(default_factory=time.time)
    last_heartbeat_at: float = field(default_factory=time.time)
    tags: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)

    def matches_capabilities(self, required: set[str]) -> bool:
        """Return True if this agent's capabilities are a superset of ``required``."""
        return required.issubset(self.capabilities)

    def capability_overlap_score(self, required: set[str]) -> float:
        """Return the fraction of required capabilities this agent satisfies.

        Returns
        -------
        float
            Value in [0.0, 1.0].  1.0 means all required capabilities are met.
            0.0 means no overlap or ``required`` is empty.
        """
        if not required:
            return 0.0
        return len(required & self.capabilities) / len(required)


class AgentNotFoundError(KeyError):
    """Raised when an agent ID is not present in the RoutingTable."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        super().__init__(f"Agent {agent_id!r} is not registered in the routing table.")


class AgentAlreadyRegisteredError(ValueError):
    """Raised when attempting to register an agent ID that already exists."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        super().__init__(
            f"Agent {agent_id!r} is already registered. "
            "Use update_agent() to modify existing records."
        )


class RoutingTable:
    """Thread-safe registry mapping agent IDs to their routing metadata.

    Parameters
    ----------
    default_healthy_only:
        When True, ``list_agents`` and ``find_by_capability`` filter out
        unhealthy agents unless explicitly overridden.
    """

    def __init__(self, *, default_healthy_only: bool = True) -> None:
        self._default_healthy_only = default_healthy_only
        self._agents: dict[str, AgentRecord] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_agent(
        self,
        agent_id: str,
        *,
        capabilities: list[str] | set[str] | None = None,
        cost_per_token: float = 0.0,
        current_load: float = 0.0,
        tags: dict[str, str] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> AgentRecord:
        """Register a new agent in the routing table.

        Parameters
        ----------
        agent_id:
            Unique identifier for the agent.
        capabilities:
            List or set of capability strings.  Defaults to empty set.
        cost_per_token:
            Estimated USD cost per LLM token.
        current_load:
            Initial load factor in [0.0, 1.0].
        tags:
            Arbitrary string key→value metadata.
        metadata:
            Additional extended attributes.

        Returns
        -------
        AgentRecord
            The newly created record.

        Raises
        ------
        AgentAlreadyRegisteredError
            If ``agent_id`` is already present.
        ValueError
            If ``current_load`` is outside [0.0, 1.0] or ``cost_per_token`` < 0.
        """
        self._validate_load(current_load)
        self._validate_cost(cost_per_token)

        with self._lock:
            if agent_id in self._agents:
                raise AgentAlreadyRegisteredError(agent_id)
            record = AgentRecord(
                agent_id=agent_id,
                capabilities=set(capabilities) if capabilities else set(),
                cost_per_token=cost_per_token,
                current_load=current_load,
                tags=dict(tags) if tags else {},
                metadata=dict(metadata) if metadata else {},
            )
            self._agents[agent_id] = record
            return record

    def deregister_agent(self, agent_id: str) -> None:
        """Remove an agent from the table.

        Raises
        ------
        AgentNotFoundError
            If ``agent_id`` is not registered.
        """
        with self._lock:
            if agent_id not in self._agents:
                raise AgentNotFoundError(agent_id)
            del self._agents[agent_id]

    def upsert_agent(
        self,
        agent_id: str,
        *,
        capabilities: list[str] | set[str] | None = None,
        cost_per_token: float = 0.0,
        current_load: float = 0.0,
        tags: dict[str, str] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> AgentRecord:
        """Register or update an agent record atomically.

        If the agent is not yet registered, it is created.  If it is already
        registered, its mutable fields are updated in-place.

        Returns
        -------
        AgentRecord
            The current (potentially updated) record.
        """
        with self._lock:
            if agent_id in self._agents:
                return self.update_agent(
                    agent_id,
                    capabilities=capabilities,
                    cost_per_token=cost_per_token,
                    current_load=current_load,
                    tags=tags,
                    metadata=metadata,
                )
            return self.register_agent(
                agent_id,
                capabilities=capabilities,
                cost_per_token=cost_per_token,
                current_load=current_load,
                tags=tags,
                metadata=metadata,
            )

    def update_agent(
        self,
        agent_id: str,
        *,
        capabilities: list[str] | set[str] | None = None,
        cost_per_token: float | None = None,
        current_load: float | None = None,
        healthy: bool | None = None,
        tags: dict[str, str] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> AgentRecord:
        """Update mutable fields on an existing agent record.

        Only non-None arguments are applied.  At least one argument must
        differ from the current value to be useful, but no error is raised if
        values are unchanged.

        Returns
        -------
        AgentRecord
            The updated record (same object, mutated in-place).

        Raises
        ------
        AgentNotFoundError
            If ``agent_id`` is not registered.
        """
        with self._lock:
            record = self._get_or_raise(agent_id)
            if capabilities is not None:
                record.capabilities = set(capabilities)
            if cost_per_token is not None:
                self._validate_cost(cost_per_token)
                record.cost_per_token = cost_per_token
            if current_load is not None:
                self._validate_load(current_load)
                record.current_load = current_load
            if healthy is not None:
                record.healthy = healthy
            if tags is not None:
                record.tags.update(tags)
            if metadata is not None:
                record.metadata.update(metadata)
            return record

    def record_heartbeat(self, agent_id: str) -> None:
        """Update the ``last_heartbeat_at`` timestamp for an agent.

        Raises
        ------
        AgentNotFoundError
            If ``agent_id`` is not registered.
        """
        with self._lock:
            record = self._get_or_raise(agent_id)
            record.last_heartbeat_at = time.time()

    def update_load(self, agent_id: str, load: float) -> None:
        """Convenience wrapper to update only the load factor.

        Parameters
        ----------
        agent_id:
            Target agent.
        load:
            New load factor in [0.0, 1.0].

        Raises
        ------
        AgentNotFoundError
            If ``agent_id`` is not registered.
        ValueError
            If ``load`` is outside [0.0, 1.0].
        """
        self.update_agent(agent_id, current_load=load)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_agent(self, agent_id: str) -> AgentRecord:
        """Return the record for ``agent_id``.

        Raises
        ------
        AgentNotFoundError
            If not registered.
        """
        with self._lock:
            return self._get_or_raise(agent_id)

    def list_agents(self, *, healthy_only: bool | None = None) -> list[AgentRecord]:
        """Return all agent records, optionally filtered to healthy-only.

        Parameters
        ----------
        healthy_only:
            If True, exclude unhealthy agents.  If None, uses the table's
            ``default_healthy_only`` setting.

        Returns
        -------
        list[AgentRecord]
            Snapshot list; mutations to returned records affect the table.
        """
        filter_healthy = (
            self._default_healthy_only if healthy_only is None else healthy_only
        )
        with self._lock:
            if filter_healthy:
                return [r for r in self._agents.values() if r.healthy]
            return list(self._agents.values())

    def find_by_capability(
        self,
        capability: str,
        *,
        healthy_only: bool | None = None,
    ) -> list[AgentRecord]:
        """Return all agents that have ``capability`` in their capability set.

        Parameters
        ----------
        capability:
            Single capability string to search for.
        healthy_only:
            Defaults to the table's ``default_healthy_only`` setting.

        Returns
        -------
        list[AgentRecord]
            Matching records, unordered.
        """
        return [
            record
            for record in self.list_agents(healthy_only=healthy_only)
            if capability in record.capabilities
        ]

    def find_by_capabilities(
        self,
        required_capabilities: set[str],
        *,
        healthy_only: bool | None = None,
        require_all: bool = True,
    ) -> list[AgentRecord]:
        """Return agents matching a set of capability requirements.

        Parameters
        ----------
        required_capabilities:
            Set of capability strings that must all be present (when
            ``require_all=True``) or at least one must be present
            (when ``require_all=False``).
        healthy_only:
            Defaults to the table's ``default_healthy_only`` setting.
        require_all:
            When True (default), an agent must have ALL capabilities.
            When False, any single capability match is sufficient.

        Returns
        -------
        list[AgentRecord]
            Matching records, unordered.
        """
        records = self.list_agents(healthy_only=healthy_only)
        if require_all:
            return [r for r in records if r.matches_capabilities(required_capabilities)]
        return [r for r in records if required_capabilities & r.capabilities]

    def __len__(self) -> int:
        with self._lock:
            return len(self._agents)

    def __contains__(self, agent_id: object) -> bool:
        with self._lock:
            return agent_id in self._agents

    def __repr__(self) -> str:
        with self._lock:
            return f"RoutingTable(agents={list(self._agents.keys())!r})"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_raise(self, agent_id: str) -> AgentRecord:
        """Return the record or raise AgentNotFoundError. Must hold _lock."""
        try:
            return self._agents[agent_id]
        except KeyError:
            raise AgentNotFoundError(agent_id) from None

    @staticmethod
    def _validate_load(load: float) -> None:
        if not (0.0 <= load <= 1.0):
            raise ValueError(
                f"current_load must be in [0.0, 1.0], got {load}."
            )

    @staticmethod
    def _validate_cost(cost: float) -> None:
        if cost < 0:
            raise ValueError(
                f"cost_per_token must be non-negative, got {cost}."
            )
