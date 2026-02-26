"""Unit tests for agent_mesh_router.messages.types."""
from __future__ import annotations

import time

import pytest

from agent_mesh_router.messages.types import HandoffMetrics, MessageType, Priority


class TestMessageType:
    def test_all_variants_exist(self) -> None:
        expected = {
            "TASK", "QUERY", "RESPONSE", "RESULT",
            "HANDOFF", "BROADCAST", "HEARTBEAT", "ERROR", "CANCEL", "ACK",
        }
        actual = {m.name for m in MessageType}
        assert actual == expected

    def test_values_are_lowercase_strings(self) -> None:
        for member in MessageType:
            assert member.value == member.value.lower()
            assert isinstance(member.value, str)

    def test_task_value(self) -> None:
        assert MessageType.TASK.value == "task"

    def test_query_value(self) -> None:
        assert MessageType.QUERY.value == "query"

    def test_response_value(self) -> None:
        assert MessageType.RESPONSE.value == "response"

    def test_result_value(self) -> None:
        assert MessageType.RESULT.value == "result"

    def test_handoff_value(self) -> None:
        assert MessageType.HANDOFF.value == "handoff"

    def test_broadcast_value(self) -> None:
        assert MessageType.BROADCAST.value == "broadcast"

    def test_heartbeat_value(self) -> None:
        assert MessageType.HEARTBEAT.value == "heartbeat"

    def test_error_value(self) -> None:
        assert MessageType.ERROR.value == "error"

    def test_cancel_value(self) -> None:
        assert MessageType.CANCEL.value == "cancel"

    def test_ack_value(self) -> None:
        assert MessageType.ACK.value == "ack"

    def test_is_str_enum(self) -> None:
        assert isinstance(MessageType.TASK, str)

    def test_json_serializable(self) -> None:
        import json
        serialized = json.dumps(MessageType.TASK.value)
        assert serialized == '"task"'

    def test_from_string_value(self) -> None:
        assert MessageType("task") is MessageType.TASK
        assert MessageType("ack") is MessageType.ACK

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            MessageType("unknown_type")


class TestPriority:
    def test_all_variants_exist(self) -> None:
        expected = {"CRITICAL", "HIGH", "NORMAL", "LOW", "BATCH"}
        assert {p.name for p in Priority} == expected

    def test_critical_is_lowest_integer(self) -> None:
        assert Priority.CRITICAL == 0

    def test_batch_is_highest_integer(self) -> None:
        assert Priority.BATCH == 4

    def test_ordering_is_correct(self) -> None:
        assert Priority.CRITICAL < Priority.HIGH < Priority.NORMAL < Priority.LOW < Priority.BATCH

    def test_high_value(self) -> None:
        assert Priority.HIGH == 1

    def test_normal_value(self) -> None:
        assert Priority.NORMAL == 2

    def test_low_value(self) -> None:
        assert Priority.LOW == 3

    def test_is_int_enum(self) -> None:
        assert isinstance(Priority.NORMAL, int)

    def test_sortable(self) -> None:
        priorities = [Priority.BATCH, Priority.CRITICAL, Priority.NORMAL]
        assert sorted(priorities) == [Priority.CRITICAL, Priority.NORMAL, Priority.BATCH]

    def test_from_integer_value(self) -> None:
        assert Priority(0) is Priority.CRITICAL
        assert Priority(4) is Priority.BATCH


class TestHandoffMetrics:
    def test_construction_with_required_fields(self) -> None:
        metrics = HandoffMetrics(
            source_agent="agent-a",
            target_agent="agent-b",
            reason="load_balancing",
        )
        assert metrics.source_agent == "agent-a"
        assert metrics.target_agent == "agent-b"
        assert metrics.reason == "load_balancing"

    def test_defaults_are_applied(self) -> None:
        metrics = HandoffMetrics(
            source_agent="a", target_agent="b", reason="test"
        )
        assert metrics.cost_incurred_usd == 0.0
        assert metrics.tokens_consumed == 0
        assert metrics.latency_ms == 0.0
        assert metrics.attempt_number == 1

    def test_handoff_timestamp_defaults_to_now(self) -> None:
        before = time.time()
        metrics = HandoffMetrics(source_agent="a", target_agent="b", reason="test")
        after = time.time()
        assert before <= metrics.handoff_timestamp <= after

    def test_custom_fields_accepted(self) -> None:
        metrics = HandoffMetrics(
            source_agent="a",
            target_agent="b",
            reason="capability_mismatch",
            cost_incurred_usd=0.0042,
            tokens_consumed=1500,
            latency_ms=250.5,
            attempt_number=3,
        )
        assert metrics.cost_incurred_usd == pytest.approx(0.0042)
        assert metrics.tokens_consumed == 1500
        assert metrics.latency_ms == pytest.approx(250.5)
        assert metrics.attempt_number == 3

    def test_total_cost_label_format(self) -> None:
        metrics = HandoffMetrics(
            source_agent="a", target_agent="b", reason="test",
            cost_incurred_usd=0.001234,
        )
        label = metrics.total_cost_label()
        assert label.startswith("$")
        assert "USD" in label
        assert "0.001234" in label

    def test_total_cost_label_zero(self) -> None:
        metrics = HandoffMetrics(source_agent="a", target_agent="b", reason="test")
        assert metrics.total_cost_label() == "$0.000000 USD"
