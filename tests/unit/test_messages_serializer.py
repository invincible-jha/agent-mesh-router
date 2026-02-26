"""Unit tests for agent_mesh_router.messages.serializer."""
from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from agent_mesh_router.messages.envelope import MessageEnvelope
from agent_mesh_router.messages.serializer import MessageSerializer, _MSGPACK_HEADER
from agent_mesh_router.messages.types import MessageType, Priority


@pytest.fixture()
def sample_envelope() -> MessageEnvelope:
    return MessageEnvelope(
        sender="service-a",
        receiver="service-b",
        payload={"command": "run", "args": [1, 2, 3]},
        message_type=MessageType.TASK,
        priority=Priority.HIGH,
        topic="tasks",
        ttl_seconds=120.0,
        cost_budget_usd=0.10,
        metadata={"region": "us-east-1"},
    )


class TestJsonEncoding:
    def test_encode_returns_bytes(self, sample_envelope: MessageEnvelope) -> None:
        result = MessageSerializer.encode(sample_envelope, format="json")
        assert isinstance(result, bytes)

    def test_encode_decode_round_trip(self, sample_envelope: MessageEnvelope) -> None:
        raw = MessageSerializer.encode(sample_envelope, format="json")
        restored = MessageSerializer.decode(raw, format="json")
        assert restored.message_id == sample_envelope.message_id
        assert restored.sender == sample_envelope.sender
        assert restored.payload == sample_envelope.payload

    def test_default_format_is_json(self, sample_envelope: MessageEnvelope) -> None:
        raw = MessageSerializer.encode(sample_envelope)
        restored = MessageSerializer.decode(raw)
        assert restored.message_id == sample_envelope.message_id

    def test_json_round_trip_preserves_enums(self, sample_envelope: MessageEnvelope) -> None:
        raw = MessageSerializer.encode(sample_envelope)
        restored = MessageSerializer.decode(raw)
        assert restored.message_type is MessageType.TASK
        assert restored.priority is Priority.HIGH

    def test_json_round_trip_with_none_fields(self) -> None:
        env = MessageEnvelope(sender="a", receiver="b", payload={})
        raw = MessageSerializer.encode(env)
        restored = MessageSerializer.decode(raw)
        assert restored.topic is None
        assert restored.parent_message_id is None
        assert restored.ttl_seconds is None
        assert restored.cost_budget_usd is None


class TestMsgpackFallback:
    def test_msgpack_fallback_when_not_installed(
        self, sample_envelope: MessageEnvelope
    ) -> None:
        with patch.dict(sys.modules, {"msgpack": None}):
            raw = MessageSerializer.encode(sample_envelope, format="msgpack")
        assert raw.startswith(_MSGPACK_HEADER)

    def test_msgpack_fallback_round_trip(self, sample_envelope: MessageEnvelope) -> None:
        with patch.dict(sys.modules, {"msgpack": None}):
            raw = MessageSerializer.encode(sample_envelope, format="msgpack")
            restored = MessageSerializer.decode(raw, format="msgpack")
        assert restored.message_id == sample_envelope.message_id

    def test_json_fallback_sentinel_detected(
        self, sample_envelope: MessageEnvelope
    ) -> None:
        json_bytes = sample_envelope.to_json()
        raw_with_sentinel = _MSGPACK_HEADER + json_bytes
        restored = MessageSerializer._decode_msgpack(raw_with_sentinel)
        assert restored.message_id == sample_envelope.message_id


class TestSupportsMsgpack:
    def test_supports_msgpack_returns_bool(self) -> None:
        result = MessageSerializer.supports_msgpack()
        assert isinstance(result, bool)

    def test_returns_false_when_not_installed(self) -> None:
        with patch.dict(sys.modules, {"msgpack": None}):
            result = MessageSerializer.supports_msgpack()
        assert result is False


class TestWireDictConversion:
    def test_envelope_to_wire_dict_has_all_keys(
        self, sample_envelope: MessageEnvelope
    ) -> None:
        wire = MessageSerializer._envelope_to_wire_dict(sample_envelope)
        required_keys = {
            "sender", "receiver", "payload", "message_type", "priority",
            "topic", "trace_id", "parent_message_id", "message_id",
            "created_at", "ttl_seconds", "cost_budget_usd", "metadata",
        }
        assert required_keys.issubset(wire.keys())

    def test_wire_dict_to_envelope_round_trip(
        self, sample_envelope: MessageEnvelope
    ) -> None:
        wire = MessageSerializer._envelope_to_wire_dict(sample_envelope)
        restored = MessageSerializer._wire_dict_to_envelope(wire)
        assert restored.message_id == sample_envelope.message_id
        assert restored.message_type is sample_envelope.message_type
