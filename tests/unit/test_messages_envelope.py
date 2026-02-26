"""Unit tests for agent_mesh_router.messages.envelope."""
from __future__ import annotations

import json
import time

import pytest

from agent_mesh_router.messages.envelope import MessageEnvelope
from agent_mesh_router.messages.types import HandoffMetrics, MessageType, Priority


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def basic_envelope() -> MessageEnvelope:
    return MessageEnvelope(
        sender="agent-a",
        receiver="agent-b",
        payload={"task": "summarize", "text": "hello world"},
    )


@pytest.fixture()
def full_envelope() -> MessageEnvelope:
    return MessageEnvelope(
        sender="agent-a",
        receiver="agent-b",
        payload={"data": 42},
        message_type=MessageType.QUERY,
        priority=Priority.HIGH,
        topic="research",
        ttl_seconds=60.0,
        cost_budget_usd=0.05,
        metadata={"env": "test"},
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestMessageEnvelopeConstruction:
    def test_required_fields_only(self, basic_envelope: MessageEnvelope) -> None:
        assert basic_envelope.sender == "agent-a"
        assert basic_envelope.receiver == "agent-b"
        assert basic_envelope.payload == {"task": "summarize", "text": "hello world"}

    def test_default_message_type_is_task(self, basic_envelope: MessageEnvelope) -> None:
        assert basic_envelope.message_type is MessageType.TASK

    def test_default_priority_is_normal(self, basic_envelope: MessageEnvelope) -> None:
        assert basic_envelope.priority is Priority.NORMAL

    def test_auto_generated_message_id(self, basic_envelope: MessageEnvelope) -> None:
        assert basic_envelope.message_id
        assert len(basic_envelope.message_id) == 36  # UUID4 format

    def test_auto_generated_trace_id(self, basic_envelope: MessageEnvelope) -> None:
        assert basic_envelope.trace_id
        assert len(basic_envelope.trace_id) == 36

    def test_unique_message_ids(self) -> None:
        env1 = MessageEnvelope(sender="a", receiver="b", payload={})
        env2 = MessageEnvelope(sender="a", receiver="b", payload={})
        assert env1.message_id != env2.message_id

    def test_unique_trace_ids(self) -> None:
        env1 = MessageEnvelope(sender="a", receiver="b", payload={})
        env2 = MessageEnvelope(sender="a", receiver="b", payload={})
        assert env1.trace_id != env2.trace_id

    def test_created_at_is_recent(self, basic_envelope: MessageEnvelope) -> None:
        assert time.time() - basic_envelope.created_at < 1.0

    def test_optional_fields_default_to_none(self, basic_envelope: MessageEnvelope) -> None:
        assert basic_envelope.topic is None
        assert basic_envelope.parent_message_id is None
        assert basic_envelope.ttl_seconds is None
        assert basic_envelope.cost_budget_usd is None

    def test_metadata_defaults_to_empty_dict(self, basic_envelope: MessageEnvelope) -> None:
        assert basic_envelope.metadata == {}

    def test_full_construction(self, full_envelope: MessageEnvelope) -> None:
        assert full_envelope.message_type is MessageType.QUERY
        assert full_envelope.priority is Priority.HIGH
        assert full_envelope.topic == "research"
        assert full_envelope.ttl_seconds == 60.0
        assert full_envelope.cost_budget_usd == 0.05
        assert full_envelope.metadata == {"env": "test"}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestMessageEnvelopeValidation:
    def test_valid_envelope_does_not_raise(self, basic_envelope: MessageEnvelope) -> None:
        basic_envelope.validate()  # Should not raise

    def test_empty_sender_raises_value_error(self) -> None:
        env = MessageEnvelope(sender="", receiver="b", payload={})
        with pytest.raises(ValueError, match="sender"):
            env.validate()

    def test_whitespace_sender_raises_value_error(self) -> None:
        env = MessageEnvelope(sender="   ", receiver="b", payload={})
        with pytest.raises(ValueError, match="sender"):
            env.validate()

    def test_empty_receiver_raises_value_error(self) -> None:
        env = MessageEnvelope(sender="a", receiver="", payload={})
        with pytest.raises(ValueError, match="receiver"):
            env.validate()

    def test_non_dict_payload_raises_type_error(self) -> None:
        env = MessageEnvelope(sender="a", receiver="b", payload={})  # type: ignore[arg-type]
        env.payload = "not-a-dict"  # type: ignore[assignment]
        with pytest.raises(TypeError, match="payload"):
            env.validate()

    def test_non_dict_metadata_raises_type_error(self) -> None:
        env = MessageEnvelope(sender="a", receiver="b", payload={})
        env.metadata = "bad"  # type: ignore[assignment]
        with pytest.raises(TypeError, match="metadata"):
            env.validate()

    def test_invalid_message_type_raises_type_error(self) -> None:
        env = MessageEnvelope(sender="a", receiver="b", payload={})
        env.message_type = "task"  # type: ignore[assignment]
        with pytest.raises(TypeError, match="message_type"):
            env.validate()

    def test_invalid_priority_raises_type_error(self) -> None:
        env = MessageEnvelope(sender="a", receiver="b", payload={})
        env.priority = 2  # type: ignore[assignment]
        with pytest.raises(TypeError, match="priority"):
            env.validate()

    def test_zero_ttl_raises_value_error(self) -> None:
        env = MessageEnvelope(sender="a", receiver="b", payload={}, ttl_seconds=0.0)
        with pytest.raises(ValueError, match="ttl_seconds"):
            env.validate()

    def test_negative_ttl_raises_value_error(self) -> None:
        env = MessageEnvelope(sender="a", receiver="b", payload={}, ttl_seconds=-5.0)
        with pytest.raises(ValueError, match="ttl_seconds"):
            env.validate()

    def test_positive_ttl_is_valid(self) -> None:
        env = MessageEnvelope(sender="a", receiver="b", payload={}, ttl_seconds=30.0)
        env.validate()  # Should not raise

    def test_negative_cost_budget_raises_value_error(self) -> None:
        env = MessageEnvelope(sender="a", receiver="b", payload={}, cost_budget_usd=-0.01)
        with pytest.raises(ValueError, match="cost_budget_usd"):
            env.validate()

    def test_zero_cost_budget_is_valid(self) -> None:
        env = MessageEnvelope(sender="a", receiver="b", payload={}, cost_budget_usd=0.0)
        env.validate()


# ---------------------------------------------------------------------------
# TTL / Expiry
# ---------------------------------------------------------------------------


class TestMessageEnvelopeExpiry:
    def test_no_ttl_never_expires(self, basic_envelope: MessageEnvelope) -> None:
        assert basic_envelope.is_expired() is False

    def test_future_ttl_not_expired(self) -> None:
        env = MessageEnvelope(sender="a", receiver="b", payload={}, ttl_seconds=3600.0)
        assert env.is_expired() is False

    def test_elapsed_ttl_is_expired(self) -> None:
        env = MessageEnvelope(sender="a", receiver="b", payload={}, ttl_seconds=1.0)
        env.created_at = time.time() - 10.0  # Pretend created 10 s ago
        assert env.is_expired() is True


# ---------------------------------------------------------------------------
# Factory methods
# ---------------------------------------------------------------------------


class TestCreateResponse:
    def test_create_response_flips_sender_receiver(self, basic_envelope: MessageEnvelope) -> None:
        reply = basic_envelope.create_response({"status": "ok"})
        assert reply.sender == basic_envelope.receiver
        assert reply.receiver == basic_envelope.sender

    def test_create_response_inherits_trace_id(self, basic_envelope: MessageEnvelope) -> None:
        reply = basic_envelope.create_response({})
        assert reply.trace_id == basic_envelope.trace_id

    def test_create_response_sets_parent_message_id(self, basic_envelope: MessageEnvelope) -> None:
        reply = basic_envelope.create_response({})
        assert reply.parent_message_id == basic_envelope.message_id

    def test_create_response_default_type_is_response(
        self, basic_envelope: MessageEnvelope
    ) -> None:
        reply = basic_envelope.create_response({})
        assert reply.message_type is MessageType.RESPONSE

    def test_create_response_custom_type(self, basic_envelope: MessageEnvelope) -> None:
        reply = basic_envelope.create_response({}, message_type=MessageType.RESULT)
        assert reply.message_type is MessageType.RESULT

    def test_create_response_inherits_priority(self, full_envelope: MessageEnvelope) -> None:
        reply = full_envelope.create_response({})
        assert reply.priority is full_envelope.priority

    def test_create_response_custom_priority(self, basic_envelope: MessageEnvelope) -> None:
        reply = basic_envelope.create_response({}, priority=Priority.CRITICAL)
        assert reply.priority is Priority.CRITICAL

    def test_create_response_has_new_message_id(self, basic_envelope: MessageEnvelope) -> None:
        reply = basic_envelope.create_response({})
        assert reply.message_id != basic_envelope.message_id

    def test_create_response_carries_payload(self, basic_envelope: MessageEnvelope) -> None:
        reply = basic_envelope.create_response({"answer": 42})
        assert reply.payload == {"answer": 42}


class TestCreateHandoff:
    def test_handoff_has_handoff_type(self, basic_envelope: MessageEnvelope) -> None:
        metrics = HandoffMetrics(
            source_agent="agent-b",
            target_agent="agent-c",
            reason="capability_mismatch",
        )
        handoff = basic_envelope.create_handoff("agent-c", metrics)
        assert handoff.message_type is MessageType.HANDOFF

    def test_handoff_receiver_is_new_agent(self, basic_envelope: MessageEnvelope) -> None:
        metrics = HandoffMetrics(
            source_agent="agent-b", target_agent="agent-c", reason="load"
        )
        handoff = basic_envelope.create_handoff("agent-c", metrics)
        assert handoff.receiver == "agent-c"

    def test_handoff_embeds_metrics_in_payload(self, basic_envelope: MessageEnvelope) -> None:
        metrics = HandoffMetrics(
            source_agent="agent-b",
            target_agent="agent-c",
            reason="load_balancing",
            cost_incurred_usd=0.01,
        )
        handoff = basic_envelope.create_handoff("agent-c", metrics)
        assert "__handoff_metrics__" in handoff.payload
        embedded = handoff.payload["__handoff_metrics__"]
        assert embedded["reason"] == "load_balancing"  # type: ignore[index]

    def test_handoff_inherits_trace_id(self, basic_envelope: MessageEnvelope) -> None:
        metrics = HandoffMetrics(source_agent="b", target_agent="c", reason="x")
        handoff = basic_envelope.create_handoff("c", metrics)
        assert handoff.trace_id == basic_envelope.trace_id

    def test_handoff_with_updated_payload(self, basic_envelope: MessageEnvelope) -> None:
        metrics = HandoffMetrics(source_agent="b", target_agent="c", reason="x")
        handoff = basic_envelope.create_handoff("c", metrics, updated_payload={"new": "data"})
        assert handoff.payload["new"] == "data"

    def test_handoff_with_remaining_budget(self, basic_envelope: MessageEnvelope) -> None:
        basic_envelope.cost_budget_usd = 0.10
        metrics = HandoffMetrics(source_agent="b", target_agent="c", reason="x")
        handoff = basic_envelope.create_handoff(
            "c", metrics, cost_budget_remaining_usd=0.07
        )
        assert handoff.cost_budget_usd == pytest.approx(0.07)


class TestCreateAck:
    def test_ack_type_is_ack(self, basic_envelope: MessageEnvelope) -> None:
        ack = basic_envelope.create_ack()
        assert ack.message_type is MessageType.ACK

    def test_ack_priority_is_high(self, basic_envelope: MessageEnvelope) -> None:
        ack = basic_envelope.create_ack()
        assert ack.priority is Priority.HIGH

    def test_ack_payload_contains_acked_id(self, basic_envelope: MessageEnvelope) -> None:
        ack = basic_envelope.create_ack()
        assert ack.payload["acked_message_id"] == basic_envelope.message_id

    def test_ack_sender_receiver_flipped(self, basic_envelope: MessageEnvelope) -> None:
        ack = basic_envelope.create_ack()
        assert ack.sender == basic_envelope.receiver
        assert ack.receiver == basic_envelope.sender


class TestCreateError:
    def test_error_type_is_error(self, basic_envelope: MessageEnvelope) -> None:
        error = basic_envelope.create_error("something failed")
        assert error.message_type is MessageType.ERROR

    def test_error_priority_is_high(self, basic_envelope: MessageEnvelope) -> None:
        error = basic_envelope.create_error("fail")
        assert error.priority is Priority.HIGH

    def test_error_payload_contains_message(self, basic_envelope: MessageEnvelope) -> None:
        error = basic_envelope.create_error("processing failed")
        assert error.payload["error_message"] == "processing failed"

    def test_error_payload_contains_default_code(self, basic_envelope: MessageEnvelope) -> None:
        error = basic_envelope.create_error("fail")
        assert error.payload["error_code"] == "PROCESSING_ERROR"

    def test_error_payload_contains_custom_code(self, basic_envelope: MessageEnvelope) -> None:
        error = basic_envelope.create_error("fail", error_code="TIMEOUT")
        assert error.payload["error_code"] == "TIMEOUT"

    def test_error_contains_failed_message_id(self, basic_envelope: MessageEnvelope) -> None:
        error = basic_envelope.create_error("fail")
        assert error.payload["failed_message_id"] == basic_envelope.message_id


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestMessageEnvelopeSerialization:
    def test_to_json_returns_bytes(self, basic_envelope: MessageEnvelope) -> None:
        raw = basic_envelope.to_json()
        assert isinstance(raw, bytes)

    def test_to_json_is_valid_json(self, basic_envelope: MessageEnvelope) -> None:
        raw = basic_envelope.to_json()
        data = json.loads(raw)
        assert isinstance(data, dict)

    def test_json_round_trip_preserves_all_fields(
        self, full_envelope: MessageEnvelope
    ) -> None:
        raw = full_envelope.to_json()
        restored = MessageEnvelope.from_json(raw)
        assert restored.sender == full_envelope.sender
        assert restored.receiver == full_envelope.receiver
        assert restored.payload == full_envelope.payload
        assert restored.message_type == full_envelope.message_type
        assert restored.priority == full_envelope.priority
        assert restored.topic == full_envelope.topic
        assert restored.trace_id == full_envelope.trace_id
        assert restored.message_id == full_envelope.message_id
        assert restored.ttl_seconds == full_envelope.ttl_seconds
        assert restored.cost_budget_usd == full_envelope.cost_budget_usd
        assert restored.metadata == full_envelope.metadata

    def test_from_json_accepts_string(self, basic_envelope: MessageEnvelope) -> None:
        json_str = basic_envelope.to_json().decode("utf-8")
        restored = MessageEnvelope.from_json(json_str)
        assert restored.message_id == basic_envelope.message_id

    def test_from_json_invalid_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Failed to parse"):
            MessageEnvelope.from_json(b"not-json{{{")

    def test_from_json_non_object_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            MessageEnvelope.from_json(b"[1, 2, 3]")

    def test_from_json_missing_required_field_raises(self) -> None:
        minimal = {"receiver": "b", "payload": {}, "message_id": "x", "trace_id": "y"}
        with pytest.raises(ValueError, match="sender"):
            MessageEnvelope.from_json(json.dumps(minimal).encode())

    def test_from_json_invalid_message_type_raises(self) -> None:
        data = {
            "sender": "a", "receiver": "b", "payload": {},
            "message_id": "x", "trace_id": "y",
            "message_type": "invalid_type", "priority": 2,
        }
        with pytest.raises(ValueError, match="message_type"):
            MessageEnvelope.from_json(json.dumps(data).encode())

    def test_from_json_invalid_priority_raises(self) -> None:
        data = {
            "sender": "a", "receiver": "b", "payload": {},
            "message_id": "x", "trace_id": "y",
            "message_type": "task", "priority": 99,
        }
        with pytest.raises(ValueError, match="priority"):
            MessageEnvelope.from_json(json.dumps(data).encode())

    def test_to_dict_matches_json_structure(self, basic_envelope: MessageEnvelope) -> None:
        d = basic_envelope.to_dict()
        from_json = json.loads(basic_envelope.to_json())
        assert d == from_json

    def test_repr_contains_key_info(self, basic_envelope: MessageEnvelope) -> None:
        r = repr(basic_envelope)
        assert "MessageEnvelope" in r
        assert "task" in r
        assert "NORMAL" in r
