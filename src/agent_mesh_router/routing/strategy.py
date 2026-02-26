"""Routing strategy ABC and five concrete implementations.

Routing strategies encapsulate the decision logic for selecting which
agent(s) should handle a given ``MessageEnvelope``.  They are stateless
(or near-stateless) so they can be shared across many ``RouteResolver``
instances without locking concerns.

Strategies
----------
CapabilityMatchRouter
    Select agents that satisfy all capabilities listed in
    ``envelope.payload["required_capabilities"]``.

CostAwareRouter
    Select the cheapest available agent, optionally excluding agents that
    would exceed the envelope's ``cost_budget_usd``.

RoundRobinRouter
    Distribute messages evenly across all healthy agents in registration order.

LeastLoadedRouter
    Route to whichever healthy agent currently has the lowest ``current_load``.

CompositeRouter
    Combine multiple strategies using configurable weights; each strategy
    scores candidates and the composite selects the highest weighted score.

Usage
-----
::

    from agent_mesh_router.routing.strategy import (
        CapabilityMatchRouter,
        CompositeRouter,
        CostAwareRouter,
        LeastLoadedRouter,
        RoundRobinRouter,
    )

    router = CompositeRouter([
        (CapabilityMatchRouter(), 0.6),
        (LeastLoadedRouter(), 0.4),
    ])
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod

from agent_mesh_router.messages.envelope import MessageEnvelope
from agent_mesh_router.routing.table import AgentRecord, RoutingTable


class RoutingStrategy(ABC):
    """Abstract base class for all routing strategies.

    A strategy receives the full ``RoutingTable`` and the ``MessageEnvelope``
    being routed and returns an ordered list of candidate ``AgentRecord``
    objects.  The first record in the returned list is the primary candidate;
    subsequent records are fallbacks.

    The strategy MUST NOT mutate the envelope or the table.
    """

    @abstractmethod
    def rank_candidates(
        self,
        table: RoutingTable,
        envelope: MessageEnvelope,
    ) -> list[AgentRecord]:
        """Return candidate agents ordered from best to worst.

        Parameters
        ----------
        table:
            The current routing table snapshot.
        envelope:
            The message that needs to be routed.

        Returns
        -------
        list[AgentRecord]
            Ordered candidates.  Empty list means no suitable agent found.
        """

    def select(
        self,
        table: RoutingTable,
        envelope: MessageEnvelope,
    ) -> AgentRecord | None:
        """Return the best candidate or None if no agents are available.

        This is a convenience wrapper around ``rank_candidates`` that returns
        only the top result.
        """
        candidates = self.rank_candidates(table, envelope)
        return candidates[0] if candidates else None


class CapabilityMatchRouter(RoutingStrategy):
    """Route to agents that possess all required capabilities.

    Required capabilities are read from ``envelope.payload`` under the key
    ``"required_capabilities"`` (a list of strings).  If the key is absent,
    all healthy agents are candidates and are ordered by ascending load.

    Parameters
    ----------
    require_all:
        When True (default), an agent must have every listed capability.
        When False, any single capability overlap makes it a candidate.
    tie_break_by_load:
        When multiple agents have equal capability match, sort by load
        (ascending) to prefer less-busy agents.  Default True.
    """

    def __init__(
        self,
        *,
        require_all: bool = True,
        tie_break_by_load: bool = True,
    ) -> None:
        self._require_all = require_all
        self._tie_break_by_load = tie_break_by_load

    def rank_candidates(
        self,
        table: RoutingTable,
        envelope: MessageEnvelope,
    ) -> list[AgentRecord]:
        raw = envelope.payload.get("required_capabilities", [])
        required: set[str] = set(raw) if isinstance(raw, list) else set()

        if not required:
            candidates = table.list_agents(healthy_only=True)
        else:
            candidates = table.find_by_capabilities(
                required,
                healthy_only=True,
                require_all=self._require_all,
            )

        if not candidates:
            return []

        if self._tie_break_by_load:
            candidates.sort(key=lambda r: r.current_load)

        return candidates


class CostAwareRouter(RoutingStrategy):
    """Route to the cheapest available agent.

    When ``envelope.cost_budget_usd`` is set, agents whose estimated cost
    per processed token would exceed the budget at a given token threshold
    are filtered out before ranking.

    The token threshold is read from
    ``envelope.payload.get("estimated_tokens", 1000)``.

    Parameters
    ----------
    budget_safety_margin:
        Fraction of the cost budget to reserve as headroom.
        E.g. 0.1 means agents are excluded if their cost would exceed
        90 % of the budget.  Defaults to 0.0 (use full budget).
    """

    def __init__(self, *, budget_safety_margin: float = 0.0) -> None:
        if not (0.0 <= budget_safety_margin < 1.0):
            raise ValueError(
                f"budget_safety_margin must be in [0.0, 1.0), got {budget_safety_margin}."
            )
        self._budget_safety_margin = budget_safety_margin

    def rank_candidates(
        self,
        table: RoutingTable,
        envelope: MessageEnvelope,
    ) -> list[AgentRecord]:
        candidates = table.list_agents(healthy_only=True)
        if not candidates:
            return []

        budget = envelope.cost_budget_usd
        if budget is not None:
            effective_budget = budget * (1.0 - self._budget_safety_margin)
            estimated_tokens = int(envelope.payload.get("estimated_tokens", 1000))  # type: ignore[arg-type]
            candidates = [
                r
                for r in candidates
                if r.cost_per_token * estimated_tokens <= effective_budget
            ]

        if not candidates:
            return []

        # Sort ascending by cost (cheapest first).
        candidates.sort(key=lambda r: r.cost_per_token)
        return candidates


class RoundRobinRouter(RoutingStrategy):
    """Distribute messages evenly across healthy agents in registration order.

    The counter is per-instance, so each ``RoundRobinRouter`` instance
    maintains its own position.  This is thread-safe via an internal lock.
    """

    def __init__(self) -> None:
        self._counter: int = 0
        self._lock = threading.Lock()

    def rank_candidates(
        self,
        table: RoutingTable,
        envelope: MessageEnvelope,
    ) -> list[AgentRecord]:
        candidates = table.list_agents(healthy_only=True)
        if not candidates:
            return []

        # Sort deterministically so round-robin order is stable.
        candidates.sort(key=lambda r: r.agent_id)

        with self._lock:
            index = self._counter % len(candidates)
            self._counter += 1

        # Rotate candidates so the selected agent is first.
        return candidates[index:] + candidates[:index]


class LeastLoadedRouter(RoutingStrategy):
    """Route to the healthy agent with the lowest current load.

    Parameters
    ----------
    load_ceiling:
        Maximum acceptable load factor.  Agents with ``current_load``
        above this value are excluded.  Defaults to 1.0 (include all).
    """

    def __init__(self, *, load_ceiling: float = 1.0) -> None:
        if not (0.0 <= load_ceiling <= 1.0):
            raise ValueError(
                f"load_ceiling must be in [0.0, 1.0], got {load_ceiling}."
            )
        self._load_ceiling = load_ceiling

    def rank_candidates(
        self,
        table: RoutingTable,
        envelope: MessageEnvelope,
    ) -> list[AgentRecord]:
        candidates = [
            r
            for r in table.list_agents(healthy_only=True)
            if r.current_load <= self._load_ceiling
        ]
        if not candidates:
            return []
        candidates.sort(key=lambda r: r.current_load)
        return candidates


class CompositeRouter(RoutingStrategy):
    """Combine multiple strategies with configurable weights.

    Each strategy produces a ranked list; the composite assigns a score to
    each candidate based on the strategy's rank position (higher rank →
    lower score) multiplied by the strategy's weight.  Candidates are then
    sorted by total weighted score descending.

    Parameters
    ----------
    strategies:
        List of (strategy, weight) tuples.  Weights are relative and do not
        need to sum to 1.0.

    Example
    -------
    ::

        router = CompositeRouter([
            (CapabilityMatchRouter(), 0.6),
            (LeastLoadedRouter(),     0.3),
            (CostAwareRouter(),       0.1),
        ])
    """

    def __init__(self, strategies: list[tuple[RoutingStrategy, float]]) -> None:
        if not strategies:
            raise ValueError("CompositeRouter requires at least one strategy.")
        for strategy, weight in strategies:
            if weight < 0:
                raise ValueError(
                    f"Strategy weight must be non-negative, got {weight} for {strategy!r}."
                )
        self._strategies = strategies

    def rank_candidates(
        self,
        table: RoutingTable,
        envelope: MessageEnvelope,
    ) -> list[AgentRecord]:
        # Collect all unique healthy candidates across all strategies.
        all_candidates: dict[str, AgentRecord] = {}
        for record in table.list_agents(healthy_only=True):
            all_candidates[record.agent_id] = record

        if not all_candidates:
            return []

        # Accumulate weighted scores.  Score for a candidate from a given
        # strategy = (N - rank_index) / N * weight, where N is total
        # candidates in that strategy's ranking.
        scores: dict[str, float] = {agent_id: 0.0 for agent_id in all_candidates}

        for strategy, weight in self._strategies:
            ranked = strategy.rank_candidates(table, envelope)
            total = len(ranked)
            if total == 0:
                continue
            for rank_index, record in enumerate(ranked):
                if record.agent_id in scores:
                    scores[record.agent_id] += (total - rank_index) / total * weight

        sorted_candidates = sorted(
            all_candidates.values(),
            key=lambda r: scores.get(r.agent_id, 0.0),
            reverse=True,
        )
        # Filter out zero-scored candidates (excluded by all strategies).
        return [r for r in sorted_candidates if scores.get(r.agent_id, 0.0) > 0.0]
