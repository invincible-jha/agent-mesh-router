"""Tests for RouteResolver, RoutingResult, and NoRouteFoundError."""
from __future__ import annotations

import pytest

from agent_mesh_router.messages.envelope import MessageEnvelope
from agent_mesh_router.routing.resolver import (
    NoRouteFoundError,
    RouteResolver,
    RoutingResult,
)
from agent_mesh_router.routing.strategy import LeastLoadedRouter, RoundRobinRouter
from agent_mesh_router.routing.table import RoutingTable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _table(*agent_ids: str) -> RoutingTable:
    table = RoutingTable()
    for i, agent_id in enumerate(agent_ids):
        table.register_agent(agent_id, current_load=i * 0.1)
    return table


def _envelope(receiver: str = "*", payload: dict | None = None) -> MessageEnvelope:
    return MessageEnvelope(
        sender="test-sender",
        receiver=receiver,
        payload=payload or {},
    )


# ---------------------------------------------------------------------------
# RoutingResult
# ---------------------------------------------------------------------------


class TestRoutingResult:
    def test_all_candidates_with_primary(self) -> None:
        result = RoutingResult(
            primary_agent_id="a",
            fallback_agent_ids=["b", "c"],
            routed=True,
        )
        assert result.all_candidates == ["a", "b", "c"]

    def test_all_candidates_without_primary(self) -> None:
        result = RoutingResult(
            primary_agent_id=None,
            fallback_agent_ids=["b"],
            routed=False,
        )
        assert result.all_candidates == ["b"]

    def test_all_candidates_empty(self) -> None:
        result = RoutingResult(primary_agent_id=None, routed=False)
        assert result.all_candidates == []


# ---------------------------------------------------------------------------
# NoRouteFoundError
# ---------------------------------------------------------------------------


class TestNoRouteFoundError:
    def test_message_contains_relevant_info(self) -> None:
        env = _envelope()
        err = NoRouteFoundError(env, "LeastLoadedRouter")
        assert "LeastLoadedRouter" in str(err)
        assert env.message_id in str(err)

    def test_envelope_attribute(self) -> None:
        env = _envelope()
        err = NoRouteFoundError(env, "TestStrategy")
        assert err.envelope is env
        assert err.strategy_name == "TestStrategy"


# ---------------------------------------------------------------------------
# RouteResolver — basic resolution
# ---------------------------------------------------------------------------


class TestRouteResolver:
    def test_direct_routing_named_healthy_receiver(self) -> None:
        table = _table("agent-1", "agent-2")
        resolver = RouteResolver(table=table, strategy=RoundRobinRouter())
        env = _envelope(receiver="agent-1")
        result = resolver.resolve(env)
        assert result.routed is True
        assert result.primary_agent_id == "agent-1"
        assert "direct" in result.strategy_name

    def test_direct_routing_skipped_for_wildcard(self) -> None:
        table = _table("agent-1", "agent-2")
        resolver = RouteResolver(table=table, strategy=LeastLoadedRouter())
        env = _envelope(receiver="*")
        result = resolver.resolve(env)
        assert result.routed is True
        # Strategy name should not say direct.
        assert "direct" not in result.strategy_name

    def test_direct_routing_unhealthy_receiver_falls_through(self) -> None:
        table = _table("sick")
        table.update_agent("sick", healthy=False)
        # Register another healthy agent.
        table.register_agent("healthy", current_load=0.0)
        resolver = RouteResolver(table=table, strategy=LeastLoadedRouter())
        env = _envelope(receiver="sick")
        result = resolver.resolve(env)
        # sick is unhealthy, so strategy routing is used.
        assert result.routed is True
        assert result.primary_agent_id == "healthy"

    def test_direct_routing_unknown_receiver_falls_through(self) -> None:
        table = _table("real-agent")
        resolver = RouteResolver(table=table, strategy=LeastLoadedRouter())
        env = _envelope(receiver="unknown-agent")
        result = resolver.resolve(env)
        assert result.routed is True
        assert result.primary_agent_id == "real-agent"

    def test_no_route_when_no_agents(self) -> None:
        table = RoutingTable()
        resolver = RouteResolver(table=table, strategy=LeastLoadedRouter())
        result = resolver.resolve(_envelope())
        assert result.routed is False
        assert result.primary_agent_id is None
        assert result.candidates_evaluated == 0

    def test_fallback_agents_populated(self) -> None:
        table = _table("a", "b", "c", "d")
        resolver = RouteResolver(table=table, strategy=RoundRobinRouter(), max_fallbacks=2)
        result = resolver.resolve(_envelope())
        assert result.routed is True
        assert len(result.fallback_agent_ids) <= 2

    def test_strategy_name_in_result(self) -> None:
        table = _table("a")
        resolver = RouteResolver(table=table, strategy=LeastLoadedRouter())
        result = resolver.resolve(_envelope())
        assert "LeastLoadedRouter" in result.strategy_name

    def test_candidates_evaluated_count(self) -> None:
        table = _table("a", "b", "c")
        resolver = RouteResolver(table=table, strategy=LeastLoadedRouter())
        result = resolver.resolve(_envelope())
        assert result.candidates_evaluated == 3

    def test_table_property(self) -> None:
        table = _table("a")
        resolver = RouteResolver(table=table, strategy=LeastLoadedRouter())
        assert resolver.table is table

    def test_strategy_property(self) -> None:
        table = _table("a")
        strategy = LeastLoadedRouter()
        resolver = RouteResolver(table=table, strategy=strategy)
        assert resolver.strategy is strategy

    def test_log_decisions_does_not_break_resolution(self) -> None:
        table = _table("a")
        resolver = RouteResolver(
            table=table, strategy=LeastLoadedRouter(), log_decisions=True
        )
        result = resolver.resolve(_envelope())
        assert result.routed is True

    def test_log_decisions_no_route(self) -> None:
        table = RoutingTable()
        resolver = RouteResolver(
            table=table, strategy=LeastLoadedRouter(), log_decisions=True
        )
        result = resolver.resolve(_envelope())
        assert result.routed is False


# ---------------------------------------------------------------------------
# RouteResolver — resolve_or_raise
# ---------------------------------------------------------------------------


class TestResolveOrRaise:
    def test_returns_result_when_routed(self) -> None:
        table = _table("agent-1")
        resolver = RouteResolver(table=table, strategy=LeastLoadedRouter())
        result = resolver.resolve_or_raise(_envelope())
        assert result.routed is True

    def test_raises_when_no_route(self) -> None:
        table = RoutingTable()
        resolver = RouteResolver(table=table, strategy=LeastLoadedRouter())
        with pytest.raises(NoRouteFoundError):
            resolver.resolve_or_raise(_envelope())


# ---------------------------------------------------------------------------
# RouteResolver — resolve_async
# ---------------------------------------------------------------------------


class TestResolveAsync:
    @pytest.mark.asyncio
    async def test_async_resolution_succeeds(self) -> None:
        table = _table("worker-1", "worker-2")
        resolver = RouteResolver(table=table, strategy=LeastLoadedRouter())
        result = await resolver.resolve_async(_envelope())
        assert result.routed is True

    @pytest.mark.asyncio
    async def test_async_no_route(self) -> None:
        table = RoutingTable()
        resolver = RouteResolver(table=table, strategy=LeastLoadedRouter())
        result = await resolver.resolve_async(_envelope())
        assert result.routed is False
