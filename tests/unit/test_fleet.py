"""Tests for fleet: FleetRegistry, LoadBalancer, HealthMonitor."""
from __future__ import annotations

import asyncio
import time

import pytest

from agent_mesh_router.fleet.health import HealthMonitor
from agent_mesh_router.fleet.load_balancer import LoadBalancer, Strategy
from agent_mesh_router.fleet.registry import (
    AgentNode,
    AgentNodeNotFoundError,
    AgentStatus,
    FleetRegistry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node(
    agent_id: str,
    capabilities: set[str] | None = None,
    load_score: float = 0.0,
    status: AgentStatus = AgentStatus.HEALTHY,
) -> AgentNode:
    return AgentNode(
        agent_id=agent_id,
        capabilities=capabilities or set(),
        load_score=load_score,
        status=status,
    )


# ---------------------------------------------------------------------------
# FleetRegistry
# ---------------------------------------------------------------------------


class TestFleetRegistry:
    def test_register_and_get(self) -> None:
        r = FleetRegistry()
        node = _node("w1")
        r.register(node)
        assert r.get("w1") is node

    def test_register_invalid_load_score(self) -> None:
        r = FleetRegistry()
        with pytest.raises(ValueError):
            r.register(_node("x", load_score=1.5))

    def test_register_overwrites_existing(self) -> None:
        r = FleetRegistry()
        r.register(_node("w1", load_score=0.1))
        new_node = _node("w1", load_score=0.9)
        r.register(new_node)
        assert r.get("w1").load_score == 0.9

    def test_deregister(self) -> None:
        r = FleetRegistry()
        r.register(_node("w1"))
        r.deregister("w1")
        with pytest.raises(AgentNodeNotFoundError):
            r.get("w1")

    def test_deregister_nonexistent_raises(self) -> None:
        r = FleetRegistry()
        with pytest.raises(AgentNodeNotFoundError):
            r.deregister("ghost")

    def test_heartbeat_updates_timestamp(self) -> None:
        r = FleetRegistry()
        r.register(_node("w1"))
        before = r.get("w1").last_heartbeat
        time.sleep(0.01)
        r.heartbeat("w1")
        after = r.get("w1").last_heartbeat
        assert after >= before

    def test_heartbeat_updates_load_score(self) -> None:
        r = FleetRegistry()
        r.register(_node("w1"))
        r.heartbeat("w1", load_score=0.8)
        assert r.get("w1").load_score == pytest.approx(0.8)

    def test_heartbeat_updates_status(self) -> None:
        r = FleetRegistry()
        r.register(_node("w1", status=AgentStatus.UNHEALTHY))
        r.heartbeat("w1", status=AgentStatus.HEALTHY)
        assert r.get("w1").status == AgentStatus.HEALTHY

    def test_heartbeat_invalid_load_raises(self) -> None:
        r = FleetRegistry()
        r.register(_node("w1"))
        with pytest.raises(ValueError):
            r.heartbeat("w1", load_score=2.0)

    def test_heartbeat_unknown_raises(self) -> None:
        r = FleetRegistry()
        with pytest.raises(AgentNodeNotFoundError):
            r.heartbeat("ghost")

    def test_get_available_filters_by_capability_and_health(self) -> None:
        r = FleetRegistry()
        r.register(_node("w1", capabilities={"nlp"}, status=AgentStatus.HEALTHY))
        r.register(_node("w2", capabilities={"nlp"}, status=AgentStatus.UNHEALTHY))
        r.register(_node("w3", capabilities={"vision"}, status=AgentStatus.HEALTHY))
        available = r.get_available("nlp")
        assert len(available) == 1
        assert available[0].agent_id == "w1"

    def test_list_nodes_all(self) -> None:
        r = FleetRegistry()
        r.register(_node("w1"))
        r.register(_node("w2", status=AgentStatus.UNHEALTHY))
        all_nodes = r.list_nodes(healthy_only=False)
        assert len(all_nodes) == 2

    def test_list_nodes_healthy_only(self) -> None:
        r = FleetRegistry()
        r.register(_node("w1"))
        r.register(_node("w2", status=AgentStatus.UNHEALTHY))
        healthy = r.list_nodes(healthy_only=True)
        assert len(healthy) == 1
        assert healthy[0].agent_id == "w1"

    def test_prune_stale(self) -> None:
        r = FleetRegistry()
        old_node = AgentNode(agent_id="old", last_heartbeat=time.time() - 100)
        r.register(old_node)
        r.register(_node("fresh"))
        pruned = r.prune_stale(timeout=50)
        assert "old" in pruned
        assert "fresh" not in pruned

    def test_len_and_contains(self) -> None:
        r = FleetRegistry()
        r.register(_node("w1"))
        r.register(_node("w2"))
        assert len(r) == 2
        assert "w1" in r
        assert "ghost" not in r

    def test_repr(self) -> None:
        r = FleetRegistry()
        r.register(_node("w1"))
        assert "FleetRegistry" in repr(r)

    def test_agent_node_is_healthy_and_has_capability(self) -> None:
        node = _node("x", capabilities={"nlp"})
        assert node.is_healthy() is True
        assert node.has_capability("nlp") is True
        assert node.has_capability("vision") is False


# ---------------------------------------------------------------------------
# LoadBalancer
# ---------------------------------------------------------------------------


class TestLoadBalancer:
    def _registry_with(self, *nodes: AgentNode) -> FleetRegistry:
        r = FleetRegistry()
        for n in nodes:
            r.register(n)
        return r

    def test_round_robin(self) -> None:
        r = self._registry_with(_node("a"), _node("b"), _node("c"))
        lb = LoadBalancer(r, strategy=Strategy.ROUND_ROBIN)
        ids = [lb.select().agent_id for _ in range(6)]
        # Should cycle through a, b, c, a, b, c.
        assert ids == ["a", "b", "c", "a", "b", "c"]

    def test_least_loaded(self) -> None:
        r = self._registry_with(
            _node("heavy", load_score=0.9),
            _node("light", load_score=0.1),
        )
        lb = LoadBalancer(r, strategy=Strategy.LEAST_LOADED)
        assert lb.select().agent_id == "light"

    def test_random(self) -> None:
        r = self._registry_with(_node("a"), _node("b"), _node("c"))
        lb = LoadBalancer(r, strategy=Strategy.RANDOM, seed=42)
        result = lb.select()
        assert result is not None
        assert result.agent_id in {"a", "b", "c"}

    def test_capability_weighted(self) -> None:
        r = self._registry_with(
            _node("rich", capabilities={"a", "b", "c"}, load_score=0.5),
            _node("poor", capabilities={"a"}, load_score=0.1),
        )
        lb = LoadBalancer(r, strategy=Strategy.CAPABILITY_WEIGHTED)
        selected = lb.select(required_capability="a")
        # rich has more capabilities so higher score.
        assert selected.agent_id == "rich"

    def test_no_healthy_nodes_returns_none(self) -> None:
        r = self._registry_with(_node("x", status=AgentStatus.UNHEALTHY))
        lb = LoadBalancer(r, strategy=Strategy.ROUND_ROBIN)
        assert lb.select() is None

    def test_capability_filter(self) -> None:
        r = self._registry_with(
            _node("w1", capabilities={"nlp"}),
            _node("w2", capabilities={"vision"}),
        )
        lb = LoadBalancer(r, strategy=Strategy.ROUND_ROBIN)
        result = lb.select(required_capability="vision")
        assert result.agent_id == "w2"

    def test_select_many(self) -> None:
        r = self._registry_with(_node("a"), _node("b"), _node("c"))
        lb = LoadBalancer(r, strategy=Strategy.LEAST_LOADED)
        results = lb.select_many(2)
        assert len(results) == 2

    def test_select_many_count_zero(self) -> None:
        r = self._registry_with(_node("a"))
        lb = LoadBalancer(r, strategy=Strategy.ROUND_ROBIN)
        assert lb.select_many(0) == []

    def test_select_many_no_healthy(self) -> None:
        r = self._registry_with(_node("a", status=AgentStatus.UNHEALTHY))
        lb = LoadBalancer(r, strategy=Strategy.ROUND_ROBIN)
        assert lb.select_many(3) == []

    def test_select_many_random(self) -> None:
        r = self._registry_with(_node("a"), _node("b"), _node("c"))
        lb = LoadBalancer(r, strategy=Strategy.RANDOM, seed=99)
        results = lb.select_many(3)
        assert len(results) == 3

    def test_select_many_capability_weighted(self) -> None:
        r = self._registry_with(
            _node("a", capabilities={"x"}),
            _node("b", capabilities={"x", "y"}),
        )
        lb = LoadBalancer(r, strategy=Strategy.CAPABILITY_WEIGHTED)
        results = lb.select_many(2, required_capability="x")
        assert len(results) == 2


# ---------------------------------------------------------------------------
# HealthMonitor
# ---------------------------------------------------------------------------


class TestHealthMonitor:
    def test_invalid_timeout_raises(self) -> None:
        r = FleetRegistry()
        with pytest.raises(ValueError):
            HealthMonitor(r, heartbeat_timeout_seconds=0.0)

    def test_invalid_interval_raises(self) -> None:
        r = FleetRegistry()
        with pytest.raises(ValueError):
            HealthMonitor(r, check_interval_seconds=0.0)

    def test_check_all_marks_stale_unhealthy(self) -> None:
        r = FleetRegistry()
        old_node = AgentNode(agent_id="stale", last_heartbeat=time.time() - 100)
        r.register(old_node)
        monitor = HealthMonitor(r, heartbeat_timeout_seconds=10.0)
        newly_unhealthy = monitor.check_all()
        assert "stale" in newly_unhealthy
        assert r.get("stale").status == AgentStatus.UNHEALTHY
        assert monitor.total_marked_unhealthy == 1

    def test_check_all_skips_already_unhealthy(self) -> None:
        r = FleetRegistry()
        old_node = AgentNode(
            agent_id="dead",
            last_heartbeat=time.time() - 200,
            status=AgentStatus.UNHEALTHY,
        )
        r.register(old_node)
        monitor = HealthMonitor(r, heartbeat_timeout_seconds=10.0)
        newly = monitor.check_all()
        assert "dead" not in newly
        assert monitor.total_marked_unhealthy == 0

    def test_check_all_healthy_node_stays_healthy(self) -> None:
        r = FleetRegistry()
        r.register(_node("fresh"))
        monitor = HealthMonitor(r, heartbeat_timeout_seconds=60.0)
        newly = monitor.check_all()
        assert newly == []

    def test_check_node_marks_unhealthy(self) -> None:
        r = FleetRegistry()
        old_node = AgentNode(agent_id="old", last_heartbeat=time.time() - 100)
        r.register(old_node)
        monitor = HealthMonitor(r, heartbeat_timeout_seconds=10.0)
        became_unhealthy = monitor.check_node("old")
        assert became_unhealthy is True

    def test_check_node_healthy_returns_false(self) -> None:
        r = FleetRegistry()
        r.register(_node("fresh"))
        monitor = HealthMonitor(r, heartbeat_timeout_seconds=60.0)
        assert monitor.check_node("fresh") is False

    def test_check_node_unknown_raises(self) -> None:
        r = FleetRegistry()
        monitor = HealthMonitor(r)
        with pytest.raises(AgentNodeNotFoundError):
            monitor.check_node("ghost")

    def test_record_heartbeat_restores_unhealthy(self) -> None:
        r = FleetRegistry()
        r.register(_node("w1", status=AgentStatus.UNHEALTHY))
        monitor = HealthMonitor(r, restore_on_heartbeat=True)
        monitor.record_heartbeat("w1")
        assert r.get("w1").status == AgentStatus.HEALTHY
        assert monitor.total_restored == 1

    def test_record_heartbeat_no_restore_when_disabled(self) -> None:
        r = FleetRegistry()
        r.register(_node("w1", status=AgentStatus.UNHEALTHY))
        monitor = HealthMonitor(r, restore_on_heartbeat=False)
        monitor.record_heartbeat("w1")
        assert r.get("w1").status == AgentStatus.UNHEALTHY

    def test_record_heartbeat_unknown_agent_silently_ignored(self) -> None:
        r = FleetRegistry()
        monitor = HealthMonitor(r)
        monitor.record_heartbeat("ghost")  # Should not raise.

    def test_record_heartbeat_with_load_score(self) -> None:
        r = FleetRegistry()
        r.register(_node("w1"))
        monitor = HealthMonitor(r)
        monitor.record_heartbeat("w1", load_score=0.6)
        assert r.get("w1").load_score == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_start_and_stop(self) -> None:
        r = FleetRegistry()
        monitor = HealthMonitor(r, check_interval_seconds=60.0)
        await monitor.start()
        assert monitor.is_running is True
        await monitor.stop()
        assert monitor.is_running is False

    @pytest.mark.asyncio
    async def test_double_start_is_noop(self) -> None:
        r = FleetRegistry()
        monitor = HealthMonitor(r, check_interval_seconds=60.0)
        await monitor.start()
        await monitor.start()
        assert monitor.is_running
        await monitor.stop()

    @pytest.mark.asyncio
    async def test_stop_without_start_is_safe(self) -> None:
        r = FleetRegistry()
        monitor = HealthMonitor(r)
        await monitor.stop()  # Should not raise.

    def test_properties(self) -> None:
        r = FleetRegistry()
        monitor = HealthMonitor(r)
        assert monitor.total_checks == 0
        assert monitor.total_marked_unhealthy == 0
        assert monitor.total_restored == 0
