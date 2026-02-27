"""Tests for agent_mesh_router.deadlock.detector — DeadlockDetector."""
from __future__ import annotations

import threading

import pytest

from agent_mesh_router.deadlock.detector import (
    DeadlockDetector,
    DeadlockResult,
    WaitEdge,
    _canonical_cycle,
    _deduplicate_cycles,
)


# ===========================================================================
# WaitEdge
# ===========================================================================


class TestWaitEdge:
    def test_frozen(self) -> None:
        edge = WaitEdge(waiter="a", holder="b")
        with pytest.raises(Exception):
            edge.waiter = "x"  # type: ignore[misc]

    def test_str_without_resource(self) -> None:
        edge = WaitEdge(waiter="agent_a", holder="agent_b")
        assert "agent_a" in str(edge)
        assert "agent_b" in str(edge)

    def test_str_with_resource(self) -> None:
        edge = WaitEdge(waiter="a", holder="b", resource_id="lock_1")
        assert "lock_1" in str(edge)

    def test_default_resource_id(self) -> None:
        edge = WaitEdge(waiter="a", holder="b")
        assert edge.resource_id == ""


# ===========================================================================
# DeadlockResult
# ===========================================================================


class TestDeadlockResult:
    def _make_result(
        self,
        cycles: list[tuple[str, ...]],
        node_count: int = 3,
        edge_count: int = 3,
    ) -> DeadlockResult:
        from datetime import datetime, timezone

        return DeadlockResult(
            cycles=tuple(cycles),
            checked_at=datetime.now(timezone.utc),
            node_count=node_count,
            edge_count=edge_count,
        )

    def test_has_deadlock_false_when_no_cycles(self) -> None:
        result = self._make_result([])
        assert not result.has_deadlock

    def test_has_deadlock_true_when_cycles(self) -> None:
        result = self._make_result([("a", "b", "a")])
        assert result.has_deadlock

    def test_cycle_count(self) -> None:
        result = self._make_result([("a", "b", "a"), ("c", "d", "c")])
        assert result.cycle_count == 2

    def test_summary_no_deadlock(self) -> None:
        result = self._make_result([])
        summary = result.summary()
        assert "No deadlock" in summary
        assert "3 agents" in summary

    def test_summary_with_deadlock(self) -> None:
        result = self._make_result([("a", "b", "a")])
        summary = result.summary()
        assert "DEADLOCK" in summary
        assert "a -> b -> a" in summary

    def test_to_dict_keys(self) -> None:
        result = self._make_result([("a", "b", "a")])
        d = result.to_dict()
        assert "has_deadlock" in d
        assert "cycles" in d
        assert "cycle_count" in d
        assert "checked_at" in d
        assert "node_count" in d
        assert "edge_count" in d

    def test_to_dict_cycles_as_lists(self) -> None:
        result = self._make_result([("a", "b", "a")])
        d = result.to_dict()
        assert d["cycles"] == [["a", "b", "a"]]


# ===========================================================================
# DeadlockDetector — construction
# ===========================================================================


class TestDeadlockDetectorConstruction:
    def test_empty_on_init(self) -> None:
        detector = DeadlockDetector()
        assert len(detector) == 0
        assert detector.agents() == []

    def test_max_cycle_length_below_2_raises(self) -> None:
        with pytest.raises(ValueError, match="max_cycle_length"):
            DeadlockDetector(max_cycle_length=1)

    def test_add_wait_registers_agents(self) -> None:
        detector = DeadlockDetector()
        detector.add_wait("a", "b")
        assert "a" in detector
        assert "b" in detector
        assert len(detector) == 2

    def test_self_loop_raises(self) -> None:
        detector = DeadlockDetector()
        with pytest.raises(ValueError, match="self-loop|itself"):
            detector.add_wait("a", "a")

    def test_add_wait_with_resource(self) -> None:
        detector = DeadlockDetector()
        detector.add_wait("a", "b", resource_id="mutex_1")
        edges = detector.edges()
        assert len(edges) == 1
        assert edges[0].resource_id == "mutex_1"


# ===========================================================================
# DeadlockDetector — edge management
# ===========================================================================


class TestEdgeManagement:
    def test_remove_wait_returns_true_when_exists(self) -> None:
        detector = DeadlockDetector()
        detector.add_wait("a", "b")
        result = detector.remove_wait("a", "b")
        assert result is True

    def test_remove_wait_returns_false_when_missing(self) -> None:
        detector = DeadlockDetector()
        result = detector.remove_wait("a", "b")
        assert result is False

    def test_remove_wait_clears_edge(self) -> None:
        detector = DeadlockDetector()
        detector.add_wait("a", "b")
        detector.remove_wait("a", "b")
        assert detector.edges() == []

    def test_clear_agent_removes_outgoing(self) -> None:
        detector = DeadlockDetector()
        detector.add_wait("a", "b")
        detector.add_wait("a", "c")
        detector.clear_agent("a")
        # a should no longer appear as waiter
        for edge in detector.edges():
            assert edge.waiter != "a"
            assert edge.holder != "a"

    def test_clear_agent_removes_incoming(self) -> None:
        detector = DeadlockDetector()
        detector.add_wait("x", "a")
        detector.clear_agent("a")
        assert detector.edges() == []

    def test_clear_removes_all(self) -> None:
        detector = DeadlockDetector()
        detector.add_wait("a", "b")
        detector.add_wait("b", "c")
        detector.clear()
        assert len(detector) == 0
        assert detector.edges() == []


# ===========================================================================
# DeadlockDetector — detect (no deadlock cases)
# ===========================================================================


class TestDetectNoCycles:
    def test_empty_graph(self) -> None:
        detector = DeadlockDetector()
        result = detector.detect()
        assert not result.has_deadlock
        assert result.node_count == 0
        assert result.edge_count == 0

    def test_linear_chain_no_cycle(self) -> None:
        detector = DeadlockDetector()
        detector.add_wait("a", "b")
        detector.add_wait("b", "c")
        result = detector.detect()
        assert not result.has_deadlock

    def test_tree_no_cycle(self) -> None:
        detector = DeadlockDetector()
        detector.add_wait("a", "b")
        detector.add_wait("a", "c")
        detector.add_wait("b", "d")
        result = detector.detect()
        assert not result.has_deadlock

    def test_node_count_and_edge_count(self) -> None:
        detector = DeadlockDetector()
        detector.add_wait("a", "b")
        detector.add_wait("b", "c")
        result = detector.detect()
        assert result.node_count == 3
        assert result.edge_count == 2


# ===========================================================================
# DeadlockDetector — detect (deadlock cases)
# ===========================================================================


class TestDetectCycles:
    def test_simple_two_node_cycle(self) -> None:
        detector = DeadlockDetector()
        detector.add_wait("a", "b")
        detector.add_wait("b", "a")
        result = detector.detect()
        assert result.has_deadlock
        assert result.cycle_count == 1

    def test_three_node_cycle(self) -> None:
        detector = DeadlockDetector()
        detector.add_wait("a", "b")
        detector.add_wait("b", "c")
        detector.add_wait("c", "a")
        result = detector.detect()
        assert result.has_deadlock
        cycle_nodes = set(result.cycles[0])
        assert cycle_nodes >= {"a", "b", "c"}

    def test_cycle_path_closes_at_start(self) -> None:
        detector = DeadlockDetector()
        detector.add_wait("a", "b")
        detector.add_wait("b", "a")
        result = detector.detect()
        cycle = result.cycles[0]
        # Cycle should start and end at the same node
        assert cycle[0] == cycle[-1]

    def test_two_independent_cycles(self) -> None:
        detector = DeadlockDetector()
        # Cycle 1: a <-> b
        detector.add_wait("a", "b")
        detector.add_wait("b", "a")
        # Cycle 2: c <-> d
        detector.add_wait("c", "d")
        detector.add_wait("d", "c")
        result = detector.detect()
        assert result.has_deadlock
        # Each cycle involves different agents
        cycle_node_sets = [set(c) for c in result.cycles]
        for node_set in cycle_node_sets:
            assert node_set & {"a", "b"} or node_set & {"c", "d"}

    def test_remove_edge_breaks_cycle(self) -> None:
        detector = DeadlockDetector()
        detector.add_wait("a", "b")
        detector.add_wait("b", "a")
        assert detector.detect().has_deadlock
        detector.remove_wait("b", "a")
        assert not detector.detect().has_deadlock

    def test_clear_agent_breaks_cycle(self) -> None:
        detector = DeadlockDetector()
        detector.add_wait("a", "b")
        detector.add_wait("b", "c")
        detector.add_wait("c", "a")
        assert detector.detect().has_deadlock
        detector.clear_agent("c")
        assert not detector.detect().has_deadlock

    def test_four_node_cycle(self) -> None:
        detector = DeadlockDetector()
        for waiter, holder in [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")]:
            detector.add_wait(waiter, holder)
        result = detector.detect()
        assert result.has_deadlock


# ===========================================================================
# Internal helpers
# ===========================================================================


class TestInternalHelpers:
    def test_canonical_cycle_rotation(self) -> None:
        cycle = ("c", "a", "b", "c")
        canonical = _canonical_cycle(cycle)
        assert canonical[0] == "a"
        assert canonical[-1] == "a"

    def test_deduplicate_removes_rotations(self) -> None:
        cycle1 = ("a", "b", "c", "a")
        cycle2 = ("b", "c", "a", "b")  # same cycle, different start
        result = _deduplicate_cycles([cycle1, cycle2])
        assert len(result) == 1

    def test_deduplicate_keeps_distinct_cycles(self) -> None:
        cycle1 = ("a", "b", "a")
        cycle2 = ("c", "d", "c")
        result = _deduplicate_cycles([cycle1, cycle2])
        assert len(result) == 2


# ===========================================================================
# Thread safety
# ===========================================================================


class TestThreadSafety:
    def test_concurrent_add_wait(self) -> None:
        detector = DeadlockDetector()
        errors: list[Exception] = []

        def add_edges(prefix: str) -> None:
            try:
                for i in range(10):
                    detector.add_wait(f"{prefix}_{i}", f"{prefix}_{i + 1}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=add_edges, args=(f"t{j}",)) for j in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        # 5 groups * 10 edges; each adds 2 new nodes per edge except shared ones
        assert len(detector.edges()) == 50
