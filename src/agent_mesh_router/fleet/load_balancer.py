"""LoadBalancer — select agent nodes using configurable load-balancing strategies.

Four strategies are provided:

round_robin
    Distributes requests evenly in rotation, stable alphabetical order.
least_loaded
    Always selects the HEALTHY node with the lowest ``load_score``.
random
    Selects a HEALTHY node at random (uniform distribution).
capability_weighted
    Filters to nodes that have the requested capability, then selects the
    node with the highest capability overlap score (most capabilities matched)
    weighted against load.

Example
-------
::

    from agent_mesh_router.fleet.registry import FleetRegistry, AgentNode
    from agent_mesh_router.fleet.load_balancer import LoadBalancer, Strategy

    registry = FleetRegistry()
    registry.register(AgentNode("a1", capabilities={"nlp", "classify"}, load_score=0.2))
    registry.register(AgentNode("a2", capabilities={"nlp"}, load_score=0.1))

    lb = LoadBalancer(registry, strategy=Strategy.LEAST_LOADED)
    node = lb.select()
    print(node.agent_id)  # "a2" (lower load)
"""
from __future__ import annotations

import logging
import random as _random
import threading
from enum import Enum

from agent_mesh_router.fleet.registry import AgentNode, FleetRegistry

logger = logging.getLogger(__name__)


class Strategy(str, Enum):
    """Available load-balancing strategies."""

    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    RANDOM = "random"
    CAPABILITY_WEIGHTED = "capability_weighted"


class LoadBalancer:
    """Select an agent node from the fleet using a configurable strategy.

    Parameters
    ----------
    registry:
        The ``FleetRegistry`` to select nodes from.
    strategy:
        Load-balancing strategy to apply.  Defaults to ``ROUND_ROBIN``.
    seed:
        Optional integer seed for the random number generator used by the
        RANDOM strategy.  Useful for deterministic testing.
    """

    def __init__(
        self,
        registry: FleetRegistry,
        *,
        strategy: Strategy = Strategy.ROUND_ROBIN,
        seed: int | None = None,
    ) -> None:
        self._registry = registry
        self._strategy = strategy
        self._rng = _random.Random(seed)
        self._rr_counter: int = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Selection API
    # ------------------------------------------------------------------

    def select(
        self,
        *,
        required_capability: str | None = None,
    ) -> AgentNode | None:
        """Select a healthy node according to the configured strategy.

        Parameters
        ----------
        required_capability:
            When provided, only nodes that expose this capability are
            eligible candidates.  For CAPABILITY_WEIGHTED strategy this
            is required; for all others it acts as a pre-filter.

        Returns
        -------
        AgentNode | None
            The selected node, or None if no eligible healthy nodes exist.
        """
        healthy = self._registry.list_nodes(healthy_only=True)

        if required_capability is not None:
            healthy = [n for n in healthy if n.has_capability(required_capability)]

        if not healthy:
            logger.debug(
                "LoadBalancer: no healthy nodes available "
                "(capability=%r, strategy=%s).",
                required_capability,
                self._strategy.value,
            )
            return None

        match self._strategy:
            case Strategy.ROUND_ROBIN:
                return self._round_robin(healthy)
            case Strategy.LEAST_LOADED:
                return self._least_loaded(healthy)
            case Strategy.RANDOM:
                return self._random(healthy)
            case Strategy.CAPABILITY_WEIGHTED:
                return self._capability_weighted(
                    healthy, required_capability or ""
                )
            case _:
                raise ValueError(f"Unknown strategy: {self._strategy!r}")

    def select_many(
        self,
        count: int,
        *,
        required_capability: str | None = None,
    ) -> list[AgentNode]:
        """Select up to ``count`` distinct healthy nodes.

        Uses the configured strategy to rank candidates then returns the
        top ``count`` (or fewer if not enough nodes are available).

        Parameters
        ----------
        count:
            Maximum number of nodes to return.
        required_capability:
            Optional capability pre-filter.

        Returns
        -------
        list[AgentNode]
            Selected nodes, ordered by strategy preference.
        """
        if count <= 0:
            return []

        healthy = self._registry.list_nodes(healthy_only=True)
        if required_capability is not None:
            healthy = [n for n in healthy if n.has_capability(required_capability)]

        if not healthy:
            return []

        ranked = self._rank(healthy, required_capability or "")
        return ranked[:count]

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _round_robin(self, candidates: list[AgentNode]) -> AgentNode:
        # Deterministic ordering: sort by agent_id for stability
        candidates_sorted = sorted(candidates, key=lambda n: n.agent_id)
        with self._lock:
            index = self._rr_counter % len(candidates_sorted)
            self._rr_counter += 1
        return candidates_sorted[index]

    def _least_loaded(self, candidates: list[AgentNode]) -> AgentNode:
        return min(candidates, key=lambda n: n.load_score)

    def _random(self, candidates: list[AgentNode]) -> AgentNode:
        with self._lock:
            return self._rng.choice(candidates)

    def _capability_weighted(
        self,
        candidates: list[AgentNode],
        required_capability: str,
    ) -> AgentNode:
        """Select node with the best capability breadth weighted against load.

        Score = (1 + capability_count) / (1 + load_score)
        Highest score wins.
        """
        def score(node: AgentNode) -> float:
            return (1 + len(node.capabilities)) / (1 + node.load_score)

        return max(candidates, key=score)

    def _rank(
        self,
        candidates: list[AgentNode],
        required_capability: str,
    ) -> list[AgentNode]:
        """Return candidates sorted by strategy preference (best first)."""
        match self._strategy:
            case Strategy.ROUND_ROBIN:
                return sorted(candidates, key=lambda n: n.agent_id)
            case Strategy.LEAST_LOADED:
                return sorted(candidates, key=lambda n: n.load_score)
            case Strategy.RANDOM:
                with self._lock:
                    shuffled = list(candidates)
                    self._rng.shuffle(shuffled)
                return shuffled
            case Strategy.CAPABILITY_WEIGHTED:
                def score(node: AgentNode) -> float:
                    return (1 + len(node.capabilities)) / (1 + node.load_score)

                return sorted(candidates, key=score, reverse=True)
            case _:
                return candidates
