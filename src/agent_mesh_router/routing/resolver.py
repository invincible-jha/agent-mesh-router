"""RouteResolver — picks the best agent for a message using a strategy.

The resolver is the single public entry-point for routing decisions.
It combines a ``RoutingTable`` (what agents exist) with a
``RoutingStrategy`` (how to choose among them) and provides both
synchronous and async resolution paths.

Example
-------
::

    from agent_mesh_router.routing.resolver import RouteResolver
    from agent_mesh_router.routing.table import RoutingTable
    from agent_mesh_router.routing.strategy import LeastLoadedRouter
    from agent_mesh_router.messages.envelope import MessageEnvelope

    table = RoutingTable()
    table.register_agent("worker-1", capabilities=["summarize"], current_load=0.2)
    table.register_agent("worker-2", capabilities=["summarize"], current_load=0.8)

    resolver = RouteResolver(table=table, strategy=LeastLoadedRouter())
    result = resolver.resolve(envelope)
    # result.primary_agent_id == "worker-1"
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from agent_mesh_router.messages.envelope import MessageEnvelope
from agent_mesh_router.routing.strategy import RoutingStrategy
from agent_mesh_router.routing.table import AgentRecord, RoutingTable

logger = logging.getLogger(__name__)


@dataclass
class RoutingResult:
    """The outcome of a routing resolution attempt.

    Attributes
    ----------
    primary_agent_id:
        The best candidate agent ID, or None if no route was found.
    fallback_agent_ids:
        Ordered list of fallback agent IDs (second-best through last).
    strategy_name:
        Class name of the strategy that produced this result.
    candidates_evaluated:
        Total number of agents considered before selecting.
    routed:
        True if at least one candidate was found.
    """

    primary_agent_id: str | None
    fallback_agent_ids: list[str] = field(default_factory=list)
    strategy_name: str = ""
    candidates_evaluated: int = 0
    routed: bool = False

    @property
    def all_candidates(self) -> list[str]:
        """Return primary + fallbacks as an ordered list."""
        if self.primary_agent_id is None:
            return list(self.fallback_agent_ids)
        return [self.primary_agent_id, *self.fallback_agent_ids]


class NoRouteFoundError(RuntimeError):
    """Raised by ``resolve_or_raise`` when no agent can handle the envelope."""

    def __init__(self, envelope: MessageEnvelope, strategy_name: str) -> None:
        self.envelope = envelope
        self.strategy_name = strategy_name
        super().__init__(
            f"No route found for message {envelope.message_id!r} "
            f"(type={envelope.message_type.value}, "
            f"receiver={envelope.receiver!r}) "
            f"using strategy {strategy_name!r}. "
            "Check that healthy agents with matching capabilities are registered."
        )


class RouteResolver:
    """Combines a RoutingTable and a RoutingStrategy into a resolution service.

    Parameters
    ----------
    table:
        The routing table to query for agent candidates.
    strategy:
        The strategy used to rank and select agents.
    max_fallbacks:
        Maximum number of fallback agents to include in ``RoutingResult``.
        Defaults to 3.
    log_decisions:
        When True, emit DEBUG-level log entries for every resolution.
    """

    def __init__(
        self,
        table: RoutingTable,
        strategy: RoutingStrategy,
        *,
        max_fallbacks: int = 3,
        log_decisions: bool = False,
    ) -> None:
        self._table = table
        self._strategy = strategy
        self._max_fallbacks = max_fallbacks
        self._log_decisions = log_decisions

    @property
    def table(self) -> RoutingTable:
        """The routing table this resolver queries."""
        return self._table

    @property
    def strategy(self) -> RoutingStrategy:
        """The routing strategy this resolver uses."""
        return self._strategy

    def resolve(self, envelope: MessageEnvelope) -> RoutingResult:
        """Synchronously determine the best agent for ``envelope``.

        If the envelope's ``receiver`` field is set to a specific (non-wildcard)
        agent that is registered and healthy, that agent is returned directly
        without invoking the strategy.

        Parameters
        ----------
        envelope:
            The message to route.

        Returns
        -------
        RoutingResult
            Resolution outcome.  Check ``result.routed`` before using
            ``result.primary_agent_id``.
        """
        strategy_name = type(self._strategy).__name__

        # Fast path: if the receiver is explicitly named and available,
        # honour it without engaging the strategy.
        if envelope.receiver and envelope.receiver != "*":
            if envelope.receiver in self._table:
                record = self._table.get_agent(envelope.receiver)
                if record.healthy:
                    result = RoutingResult(
                        primary_agent_id=envelope.receiver,
                        strategy_name=f"direct:{strategy_name}",
                        candidates_evaluated=1,
                        routed=True,
                    )
                    self._maybe_log(envelope, result)
                    return result

        # Strategy-based routing.
        ranked: list[AgentRecord] = self._strategy.rank_candidates(
            self._table, envelope
        )

        if not ranked:
            result = RoutingResult(
                primary_agent_id=None,
                strategy_name=strategy_name,
                candidates_evaluated=0,
                routed=False,
            )
            self._maybe_log(envelope, result)
            return result

        primary = ranked[0]
        fallbacks = [r.agent_id for r in ranked[1: 1 + self._max_fallbacks]]

        result = RoutingResult(
            primary_agent_id=primary.agent_id,
            fallback_agent_ids=fallbacks,
            strategy_name=strategy_name,
            candidates_evaluated=len(ranked),
            routed=True,
        )
        self._maybe_log(envelope, result)
        return result

    async def resolve_async(self, envelope: MessageEnvelope) -> RoutingResult:
        """Async wrapper around ``resolve`` for use in async workflows.

        Delegates to the synchronous implementation via a thread-pool
        executor so CPU-bound ranking does not block the event loop.

        Parameters
        ----------
        envelope:
            The message to route.

        Returns
        -------
        RoutingResult
            Resolution outcome.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.resolve, envelope)

    def resolve_or_raise(self, envelope: MessageEnvelope) -> RoutingResult:
        """Resolve and raise ``NoRouteFoundError`` if no agent was found.

        Parameters
        ----------
        envelope:
            The message to route.

        Returns
        -------
        RoutingResult
            Resolution outcome with ``routed=True``.

        Raises
        ------
        NoRouteFoundError
            When no suitable agent is available.
        """
        result = self.resolve(envelope)
        if not result.routed:
            raise NoRouteFoundError(envelope, type(self._strategy).__name__)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_log(self, envelope: MessageEnvelope, result: RoutingResult) -> None:
        if not self._log_decisions:
            return
        if result.routed:
            logger.debug(
                "Routed message %s -> %s (strategy=%s, candidates=%d)",
                envelope.message_id[:8],
                result.primary_agent_id,
                result.strategy_name,
                result.candidates_evaluated,
            )
        else:
            logger.debug(
                "No route for message %s (strategy=%s)",
                envelope.message_id[:8],
                result.strategy_name,
            )
