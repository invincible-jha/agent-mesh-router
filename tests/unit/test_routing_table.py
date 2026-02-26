"""Unit tests for agent_mesh_router.routing.table."""
from __future__ import annotations

import pytest

from agent_mesh_router.routing.table import (
    AgentAlreadyRegisteredError,
    AgentNotFoundError,
    AgentRecord,
    RoutingTable,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def table() -> RoutingTable:
    return RoutingTable()


@pytest.fixture()
def populated_table() -> RoutingTable:
    t = RoutingTable()
    t.register_agent("agent-a", capabilities=["summarize", "translate"], cost_per_token=0.001)
    t.register_agent("agent-b", capabilities=["summarize"], cost_per_token=0.002, current_load=0.5)
    t.register_agent("agent-c", capabilities=["analyze"], cost_per_token=0.0005, current_load=0.8)
    return t


# ---------------------------------------------------------------------------
# AgentRecord
# ---------------------------------------------------------------------------


class TestAgentRecord:
    def test_matches_capabilities_exact(self) -> None:
        record = AgentRecord(
            agent_id="x", capabilities={"summarize", "translate"}
        )
        assert record.matches_capabilities({"summarize", "translate"}) is True

    def test_matches_capabilities_subset(self) -> None:
        record = AgentRecord(
            agent_id="x", capabilities={"summarize", "translate", "classify"}
        )
        assert record.matches_capabilities({"summarize"}) is True

    def test_matches_capabilities_superset_fails(self) -> None:
        record = AgentRecord(agent_id="x", capabilities={"summarize"})
        assert record.matches_capabilities({"summarize", "translate"}) is False

    def test_matches_capabilities_empty_required(self) -> None:
        record = AgentRecord(agent_id="x", capabilities={"summarize"})
        assert record.matches_capabilities(set()) is True

    def test_capability_overlap_score_full_match(self) -> None:
        record = AgentRecord(agent_id="x", capabilities={"a", "b", "c"})
        score = record.capability_overlap_score({"a", "b", "c"})
        assert score == pytest.approx(1.0)

    def test_capability_overlap_score_partial(self) -> None:
        record = AgentRecord(agent_id="x", capabilities={"a", "b"})
        score = record.capability_overlap_score({"a", "b", "c"})
        assert score == pytest.approx(2 / 3)

    def test_capability_overlap_score_no_match(self) -> None:
        record = AgentRecord(agent_id="x", capabilities={"z"})
        score = record.capability_overlap_score({"a", "b"})
        assert score == pytest.approx(0.0)

    def test_capability_overlap_score_empty_required(self) -> None:
        record = AgentRecord(agent_id="x", capabilities={"a"})
        assert record.capability_overlap_score(set()) == 0.0


# ---------------------------------------------------------------------------
# RoutingTable registration
# ---------------------------------------------------------------------------


class TestRoutingTableRegistration:
    def test_register_agent_returns_record(self, table: RoutingTable) -> None:
        record = table.register_agent("worker-1", capabilities=["task"])
        assert isinstance(record, AgentRecord)
        assert record.agent_id == "worker-1"

    def test_registered_agent_is_retrievable(self, table: RoutingTable) -> None:
        table.register_agent("worker-1", capabilities=["task"])
        retrieved = table.get_agent("worker-1")
        assert retrieved.agent_id == "worker-1"

    def test_register_with_capabilities_list(self, table: RoutingTable) -> None:
        table.register_agent("w1", capabilities=["a", "b", "c"])
        record = table.get_agent("w1")
        assert record.capabilities == {"a", "b", "c"}

    def test_register_with_capabilities_set(self, table: RoutingTable) -> None:
        table.register_agent("w1", capabilities={"x", "y"})
        record = table.get_agent("w1")
        assert record.capabilities == {"x", "y"}

    def test_register_sets_cost_per_token(self, table: RoutingTable) -> None:
        table.register_agent("w1", cost_per_token=0.0004)
        assert table.get_agent("w1").cost_per_token == pytest.approx(0.0004)

    def test_register_sets_current_load(self, table: RoutingTable) -> None:
        table.register_agent("w1", current_load=0.6)
        assert table.get_agent("w1").current_load == pytest.approx(0.6)

    def test_register_duplicate_raises(self, table: RoutingTable) -> None:
        table.register_agent("w1")
        with pytest.raises(AgentAlreadyRegisteredError) as exc_info:
            table.register_agent("w1")
        assert exc_info.value.agent_id == "w1"

    def test_register_invalid_load_raises(self, table: RoutingTable) -> None:
        with pytest.raises(ValueError):
            table.register_agent("w1", current_load=1.5)

    def test_register_negative_load_raises(self, table: RoutingTable) -> None:
        with pytest.raises(ValueError):
            table.register_agent("w1", current_load=-0.1)

    def test_register_negative_cost_raises(self, table: RoutingTable) -> None:
        with pytest.raises(ValueError):
            table.register_agent("w1", cost_per_token=-1.0)

    def test_register_with_tags(self, table: RoutingTable) -> None:
        table.register_agent("w1", tags={"region": "us-east-1"})
        assert table.get_agent("w1").tags == {"region": "us-east-1"}

    def test_deregister_removes_agent(self, table: RoutingTable) -> None:
        table.register_agent("w1")
        table.deregister_agent("w1")
        assert "w1" not in table

    def test_deregister_missing_raises(self, table: RoutingTable) -> None:
        with pytest.raises(AgentNotFoundError):
            table.deregister_agent("ghost")

    def test_upsert_creates_new_agent(self, table: RoutingTable) -> None:
        record = table.upsert_agent("w1", capabilities=["task"])
        assert "w1" in table
        assert record.capabilities == {"task"}

    def test_upsert_updates_existing_agent(self, table: RoutingTable) -> None:
        table.register_agent("w1", current_load=0.1)
        table.upsert_agent("w1", current_load=0.9)
        assert table.get_agent("w1").current_load == pytest.approx(0.9)

    def test_len_reflects_count(self, table: RoutingTable) -> None:
        assert len(table) == 0
        table.register_agent("a")
        table.register_agent("b")
        assert len(table) == 2

    def test_contains_operator(self, table: RoutingTable) -> None:
        table.register_agent("a")
        assert "a" in table
        assert "b" not in table


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


class TestRoutingTableUpdate:
    def test_update_capabilities(self, table: RoutingTable) -> None:
        table.register_agent("w1", capabilities=["old"])
        table.update_agent("w1", capabilities=["new", "fresh"])
        assert table.get_agent("w1").capabilities == {"new", "fresh"}

    def test_update_load(self, table: RoutingTable) -> None:
        table.register_agent("w1", current_load=0.2)
        table.update_agent("w1", current_load=0.8)
        assert table.get_agent("w1").current_load == pytest.approx(0.8)

    def test_update_healthy_false(self, table: RoutingTable) -> None:
        table.register_agent("w1")
        table.update_agent("w1", healthy=False)
        assert table.get_agent("w1").healthy is False

    def test_update_load_via_helper(self, table: RoutingTable) -> None:
        table.register_agent("w1", current_load=0.1)
        table.update_load("w1", 0.7)
        assert table.get_agent("w1").current_load == pytest.approx(0.7)

    def test_update_nonexistent_raises(self, table: RoutingTable) -> None:
        with pytest.raises(AgentNotFoundError):
            table.update_agent("ghost", current_load=0.5)

    def test_record_heartbeat_updates_timestamp(self, table: RoutingTable) -> None:
        import time
        table.register_agent("w1")
        old_ts = table.get_agent("w1").last_heartbeat_at
        time.sleep(0.01)
        table.record_heartbeat("w1")
        new_ts = table.get_agent("w1").last_heartbeat_at
        assert new_ts > old_ts

    def test_record_heartbeat_nonexistent_raises(self, table: RoutingTable) -> None:
        with pytest.raises(AgentNotFoundError):
            table.record_heartbeat("ghost")


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


class TestRoutingTableLookup:
    def test_list_agents_returns_all_healthy(
        self, populated_table: RoutingTable
    ) -> None:
        agents = populated_table.list_agents()
        assert len(agents) == 3

    def test_list_agents_healthy_only_excludes_unhealthy(
        self, populated_table: RoutingTable
    ) -> None:
        populated_table.update_agent("agent-b", healthy=False)
        agents = populated_table.list_agents(healthy_only=True)
        assert all(a.healthy for a in agents)
        assert len(agents) == 2

    def test_list_agents_healthy_false_includes_all(
        self, populated_table: RoutingTable
    ) -> None:
        populated_table.update_agent("agent-b", healthy=False)
        agents = populated_table.list_agents(healthy_only=False)
        assert len(agents) == 3

    def test_find_by_capability_returns_matching(
        self, populated_table: RoutingTable
    ) -> None:
        records = populated_table.find_by_capability("summarize")
        ids = {r.agent_id for r in records}
        assert ids == {"agent-a", "agent-b"}

    def test_find_by_capability_no_match_returns_empty(
        self, populated_table: RoutingTable
    ) -> None:
        records = populated_table.find_by_capability("nonexistent")
        assert records == []

    def test_find_by_capabilities_require_all(
        self, populated_table: RoutingTable
    ) -> None:
        records = populated_table.find_by_capabilities(
            {"summarize", "translate"}, require_all=True
        )
        assert len(records) == 1
        assert records[0].agent_id == "agent-a"

    def test_find_by_capabilities_any_match(
        self, populated_table: RoutingTable
    ) -> None:
        records = populated_table.find_by_capabilities(
            {"summarize", "analyze"}, require_all=False
        )
        ids = {r.agent_id for r in records}
        assert ids == {"agent-a", "agent-b", "agent-c"}

    def test_get_agent_not_found_raises(self, table: RoutingTable) -> None:
        with pytest.raises(AgentNotFoundError) as exc_info:
            table.get_agent("ghost")
        assert exc_info.value.agent_id == "ghost"

    def test_repr_contains_agent_ids(
        self, populated_table: RoutingTable
    ) -> None:
        r = repr(populated_table)
        assert "RoutingTable" in r
