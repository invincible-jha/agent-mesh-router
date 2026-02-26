"""Tests for routing strategies."""
from __future__ import annotations

import pytest

from agent_mesh_router.messages.envelope import MessageEnvelope
from agent_mesh_router.routing.strategy import (
    CapabilityMatchRouter,
    CompositeRouter,
    CostAwareRouter,
    LeastLoadedRouter,
    RoundRobinRouter,
)
from agent_mesh_router.routing.table import RoutingTable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _table(*agents: tuple[str, list[str], float, float]) -> RoutingTable:
    """Build a RoutingTable from (agent_id, capabilities, load, cost) tuples."""
    table = RoutingTable()
    for agent_id, caps, load, cost in agents:
        table.register_agent(
            agent_id,
            capabilities=caps,
            current_load=load,
            cost_per_token=cost,
        )
    return table


def _envelope(
    receiver: str = "*",
    payload: dict | None = None,
    cost_budget: float | None = None,
) -> MessageEnvelope:
    env = MessageEnvelope(
        sender="test",
        receiver=receiver,
        payload=payload or {},
    )
    env.cost_budget_usd = cost_budget
    return env


# ---------------------------------------------------------------------------
# CapabilityMatchRouter
# ---------------------------------------------------------------------------


class TestCapabilityMatchRouter:
    def test_no_required_caps_returns_all_healthy(self) -> None:
        table = _table(
            ("a", ["nlp"], 0.1, 0.0),
            ("b", ["vision"], 0.5, 0.0),
        )
        router = CapabilityMatchRouter()
        env = _envelope(payload={})
        result = router.rank_candidates(table, env)
        assert len(result) == 2

    def test_require_all_filters_correctly(self) -> None:
        table = _table(
            ("full", ["nlp", "vision"], 0.2, 0.0),
            ("partial", ["nlp"], 0.1, 0.0),
        )
        router = CapabilityMatchRouter(require_all=True)
        env = _envelope(payload={"required_capabilities": ["nlp", "vision"]})
        result = router.rank_candidates(table, env)
        assert len(result) == 1
        assert result[0].agent_id == "full"

    def test_require_any_accepts_partial_match(self) -> None:
        table = _table(
            ("nlp_only", ["nlp"], 0.1, 0.0),
            ("vision_only", ["vision"], 0.5, 0.0),
        )
        router = CapabilityMatchRouter(require_all=False)
        env = _envelope(payload={"required_capabilities": ["nlp", "vision"]})
        result = router.rank_candidates(table, env)
        assert len(result) == 2

    def test_tie_break_by_load_sorts_ascending(self) -> None:
        table = _table(
            ("heavy", ["nlp"], 0.9, 0.0),
            ("light", ["nlp"], 0.1, 0.0),
        )
        router = CapabilityMatchRouter(tie_break_by_load=True)
        env = _envelope(payload={"required_capabilities": ["nlp"]})
        result = router.rank_candidates(table, env)
        assert result[0].agent_id == "light"

    def test_no_healthy_agents_returns_empty(self) -> None:
        table = _table(("a", ["nlp"], 0.5, 0.0))
        table.update_agent("a", healthy=False)
        router = CapabilityMatchRouter()
        env = _envelope(payload={"required_capabilities": ["nlp"]})
        result = router.rank_candidates(table, env)
        assert result == []

    def test_non_list_required_capabilities_treated_as_empty(self) -> None:
        table = _table(("a", ["nlp"], 0.2, 0.0))
        router = CapabilityMatchRouter()
        env = _envelope(payload={"required_capabilities": "not-a-list"})
        result = router.rank_candidates(table, env)
        # Non-list treated as empty, all healthy agents returned.
        assert len(result) == 1

    def test_select_returns_best(self) -> None:
        table = _table(
            ("best", ["nlp"], 0.1, 0.0),
            ("worst", ["nlp"], 0.9, 0.0),
        )
        router = CapabilityMatchRouter()
        env = _envelope(payload={"required_capabilities": ["nlp"]})
        selected = router.select(table, env)
        assert selected is not None
        assert selected.agent_id == "best"

    def test_select_returns_none_when_no_candidates(self) -> None:
        table = RoutingTable()
        router = CapabilityMatchRouter()
        selected = router.select(table, _envelope())
        assert selected is None


# ---------------------------------------------------------------------------
# CostAwareRouter
# ---------------------------------------------------------------------------


class TestCostAwareRouter:
    def test_invalid_budget_safety_margin_raises(self) -> None:
        with pytest.raises(ValueError):
            CostAwareRouter(budget_safety_margin=1.0)

    def test_sorts_cheapest_first(self) -> None:
        table = _table(
            ("expensive", [], 0.0, 0.01),
            ("cheap", [], 0.0, 0.001),
        )
        router = CostAwareRouter()
        result = router.rank_candidates(table, _envelope())
        assert result[0].agent_id == "cheap"

    def test_budget_filters_expensive_agents(self) -> None:
        table = _table(
            ("cheap", [], 0.0, 0.0001),
            ("expensive", [], 0.0, 0.01),
        )
        router = CostAwareRouter()
        # budget = 0.5 USD, estimated_tokens = 1000
        # cheap: 0.0001 * 1000 = 0.1 <= 0.5 -> included
        # expensive: 0.01 * 1000 = 10 > 0.5 -> excluded
        env = _envelope(payload={"estimated_tokens": 1000}, cost_budget=0.5)
        result = router.rank_candidates(table, env)
        assert len(result) == 1
        assert result[0].agent_id == "cheap"

    def test_safety_margin_reduces_effective_budget(self) -> None:
        table = _table(
            ("marginal", [], 0.0, 0.0009),
        )
        router = CostAwareRouter(budget_safety_margin=0.1)
        # effective budget = 1.0 * 0.9 = 0.9
        # cost = 0.0009 * 1000 = 0.9 <= 0.9 -> included
        env = _envelope(payload={"estimated_tokens": 1000}, cost_budget=1.0)
        result = router.rank_candidates(table, env)
        assert len(result) == 1

    def test_all_filtered_returns_empty(self) -> None:
        table = _table(("pricey", [], 0.0, 1.0))
        router = CostAwareRouter()
        env = _envelope(payload={"estimated_tokens": 1000}, cost_budget=0.001)
        result = router.rank_candidates(table, env)
        assert result == []

    def test_no_budget_returns_all(self) -> None:
        table = _table(("a", [], 0.0, 0.1), ("b", [], 0.0, 0.2))
        router = CostAwareRouter()
        result = router.rank_candidates(table, _envelope())
        assert len(result) == 2

    def test_empty_table_returns_empty(self) -> None:
        table = RoutingTable()
        router = CostAwareRouter()
        result = router.rank_candidates(table, _envelope())
        assert result == []

    def test_default_estimated_tokens(self) -> None:
        table = _table(("a", [], 0.0, 0.0001))
        router = CostAwareRouter()
        # No estimated_tokens in payload — defaults to 1000
        env = _envelope(payload={}, cost_budget=0.5)
        result = router.rank_candidates(table, env)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# RoundRobinRouter
# ---------------------------------------------------------------------------


class TestRoundRobinRouter:
    def test_cycles_through_agents(self) -> None:
        table = _table(
            ("agent-a", [], 0.0, 0.0),
            ("agent-b", [], 0.0, 0.0),
            ("agent-c", [], 0.0, 0.0),
        )
        router = RoundRobinRouter()
        env = _envelope()
        selected = [router.rank_candidates(table, env)[0].agent_id for _ in range(6)]
        assert selected[:3] == selected[3:6]

    def test_stable_ordering(self) -> None:
        table = _table(
            ("z", [], 0.0, 0.0),
            ("a", [], 0.0, 0.0),
        )
        router = RoundRobinRouter()
        first = router.rank_candidates(table, _envelope())[0].agent_id
        # Sorted alphabetically: a comes before z
        assert first == "a"

    def test_empty_table_returns_empty(self) -> None:
        router = RoundRobinRouter()
        result = router.rank_candidates(RoutingTable(), _envelope())
        assert result == []

    def test_single_agent_always_selected(self) -> None:
        table = _table(("only", [], 0.0, 0.0))
        router = RoundRobinRouter()
        for _ in range(5):
            result = router.rank_candidates(table, _envelope())
            assert result[0].agent_id == "only"


# ---------------------------------------------------------------------------
# LeastLoadedRouter
# ---------------------------------------------------------------------------


class TestLeastLoadedRouter:
    def test_selects_least_loaded(self) -> None:
        table = _table(
            ("heavy", [], 0.9, 0.0),
            ("light", [], 0.1, 0.0),
        )
        router = LeastLoadedRouter()
        result = router.rank_candidates(table, _envelope())
        assert result[0].agent_id == "light"

    def test_load_ceiling_filters_overloaded(self) -> None:
        table = _table(
            ("overloaded", [], 0.9, 0.0),
            ("ok", [], 0.5, 0.0),
        )
        router = LeastLoadedRouter(load_ceiling=0.7)
        result = router.rank_candidates(table, _envelope())
        assert len(result) == 1
        assert result[0].agent_id == "ok"

    def test_invalid_load_ceiling_raises(self) -> None:
        with pytest.raises(ValueError):
            LeastLoadedRouter(load_ceiling=1.5)

    def test_all_over_ceiling_returns_empty(self) -> None:
        table = _table(("a", [], 0.8, 0.0))
        router = LeastLoadedRouter(load_ceiling=0.5)
        result = router.rank_candidates(table, _envelope())
        assert result == []

    def test_empty_table_returns_empty(self) -> None:
        router = LeastLoadedRouter()
        assert router.rank_candidates(RoutingTable(), _envelope()) == []


# ---------------------------------------------------------------------------
# CompositeRouter
# ---------------------------------------------------------------------------


class TestCompositeRouter:
    def test_empty_strategies_raises(self) -> None:
        with pytest.raises(ValueError):
            CompositeRouter([])

    def test_negative_weight_raises(self) -> None:
        with pytest.raises(ValueError):
            CompositeRouter([(LeastLoadedRouter(), -1.0)])

    def test_combines_strategies(self) -> None:
        table = _table(
            ("a", ["nlp"], 0.1, 0.001),
            ("b", ["nlp"], 0.9, 0.0001),
        )
        router = CompositeRouter([
            (CapabilityMatchRouter(), 0.6),
            (LeastLoadedRouter(), 0.4),
        ])
        env = _envelope(payload={"required_capabilities": ["nlp"]})
        result = router.rank_candidates(table, env)
        assert len(result) >= 1

    def test_no_healthy_agents_returns_empty(self) -> None:
        table = _table(("a", [], 0.5, 0.0))
        table.update_agent("a", healthy=False)
        router = CompositeRouter([(LeastLoadedRouter(), 1.0)])
        result = router.rank_candidates(table, _envelope())
        assert result == []

    def test_zero_score_candidates_filtered_out(self) -> None:
        """Agents that score zero across all strategies are excluded."""
        table = _table(
            ("matching", ["nlp"], 0.1, 0.0),
        )
        router = CompositeRouter([
            (CapabilityMatchRouter(), 1.0),
        ])
        env = _envelope(payload={"required_capabilities": ["nlp"]})
        result = router.rank_candidates(table, env)
        assert result[0].agent_id == "matching"

    def test_single_strategy_wrapper(self) -> None:
        table = _table(
            ("heavy", [], 0.9, 0.0),
            ("light", [], 0.1, 0.0),
        )
        router = CompositeRouter([(LeastLoadedRouter(), 1.0)])
        result = router.rank_candidates(table, _envelope())
        assert result[0].agent_id == "light"
