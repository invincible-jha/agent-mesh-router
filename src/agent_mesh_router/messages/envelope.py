"""MessageEnvelope — the universal container for all inter-agent messages.

Every message transmitted through the agent mesh is wrapped in a
``MessageEnvelope`` before being handed to the broker.  The envelope
carries routing metadata (sender, receiver, topic, trace_id), delivery
semantics (priority, TTL, cost budget), and the opaque payload that is
meaningful only to the agents involved.

Design notes
------------
- Uses ``dataclasses`` rather than Pydantic so the envelope is usable
  without installing Pydantic at the core layer.  Runtime validation is
  performed by an explicit ``validate()`` call.
- ``to_json()`` / ``from_json()`` rely only on the stdlib ``json`` module,
  keeping the hot serialization path dependency-free.
- Factory methods (``create_response``, ``create_handoff``) propagate
  trace_id automatically so distributed traces stay coherent.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Union

from agent_mesh_router.messages.types import HandoffMetrics, MessageType, Priority

# Payload values may be any JSON-serializable structure.
PayloadValue = Union[str, int, float, bool, None, list, dict]  # type: ignore[type-arg]


@dataclass
class MessageEnvelope:
    """Wrapper for all inter-agent messages.

    Parameters
    ----------
    sender:
        Unique ID of the originating agent.
    receiver:
        Unique ID of the target agent. Use ``"*"`` for broadcast messages.
    payload:
        Arbitrary JSON-serializable data meaningful to the agents.
    message_type:
        Semantic classification of the message (default: TASK).
    priority:
        Delivery priority relative to other queued messages (default: NORMAL).
    topic:
        Optional pub/sub topic string. When set, the broker routes by topic
        in addition to the explicit ``receiver`` field.
    trace_id:
        Distributed trace identifier.  Auto-generated as a UUID4 string if
        not supplied.
    parent_message_id:
        The ``message_id`` of the message this envelope is responding to.
        ``None`` for originating messages.
    message_id:
        Unique identifier for this specific envelope instance.
        Auto-generated as a UUID4 string.
    created_at:
        Unix epoch seconds (float) when the envelope was created.
        Defaults to ``time.time()`` at construction.
    ttl_seconds:
        Time-to-live in seconds. Brokers may discard envelopes whose age
        exceeds ``ttl_seconds``.  ``None`` disables TTL enforcement.
    cost_budget_usd:
        Maximum USD cost the receiving agent should spend processing this
        message.  ``None`` means unconstrained.
    metadata:
        Free-form key→value metadata for extensions and middleware.
    """

    sender: str
    receiver: str
    payload: dict[str, PayloadValue]
    message_type: MessageType = MessageType.TASK
    priority: Priority = Priority.NORMAL
    topic: str | None = None
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_message_id: str | None = None
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    ttl_seconds: float | None = None
    cost_budget_usd: float | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Perform runtime validation of envelope fields.

        Raises
        ------
        ValueError
            If any required field is blank or a numeric constraint is violated.
        TypeError
            If ``payload`` is not a dict or ``metadata`` is not a dict.
        """
        if not self.sender or not self.sender.strip():
            raise ValueError("MessageEnvelope.sender must be a non-empty string.")
        if not self.receiver or not self.receiver.strip():
            raise ValueError("MessageEnvelope.receiver must be a non-empty string.")
        if not isinstance(self.payload, dict):
            raise TypeError(
                f"MessageEnvelope.payload must be a dict, got {type(self.payload).__name__}."
            )
        if not isinstance(self.metadata, dict):
            raise TypeError(
                f"MessageEnvelope.metadata must be a dict, got {type(self.metadata).__name__}."
            )
        if not isinstance(self.message_type, MessageType):
            raise TypeError(
                f"MessageEnvelope.message_type must be a MessageType, "
                f"got {type(self.message_type).__name__}."
            )
        if not isinstance(self.priority, Priority):
            raise TypeError(
                f"MessageEnvelope.priority must be a Priority, "
                f"got {type(self.priority).__name__}."
            )
        if self.ttl_seconds is not None and self.ttl_seconds <= 0:
            raise ValueError(
                f"MessageEnvelope.ttl_seconds must be positive, got {self.ttl_seconds}."
            )
        if self.cost_budget_usd is not None and self.cost_budget_usd < 0:
            raise ValueError(
                f"MessageEnvelope.cost_budget_usd must be non-negative, "
                f"got {self.cost_budget_usd}."
            )

    def is_expired(self) -> bool:
        """Return True if this envelope's TTL has elapsed.

        Always returns False when ``ttl_seconds`` is None.
        """
        if self.ttl_seconds is None:
            return False
        return (time.time() - self.created_at) > self.ttl_seconds

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    def create_response(
        self,
        payload: dict[str, PayloadValue],
        *,
        message_type: MessageType = MessageType.RESPONSE,
        priority: Priority | None = None,
    ) -> "MessageEnvelope":
        """Create a reply envelope addressed back to this message's sender.

        The reply inherits ``trace_id`` from the original and sets
        ``parent_message_id`` to the original's ``message_id``.

        Parameters
        ----------
        payload:
            Response data to embed.
        message_type:
            Semantic type of the reply (default: RESPONSE).
        priority:
            Priority for the reply; defaults to the original's priority.

        Returns
        -------
        MessageEnvelope
            A new envelope ready for dispatch.
        """
        return MessageEnvelope(
            sender=self.receiver,
            receiver=self.sender,
            payload=payload,
            message_type=message_type,
            priority=priority if priority is not None else self.priority,
            topic=self.topic,
            trace_id=self.trace_id,
            parent_message_id=self.message_id,
            cost_budget_usd=self.cost_budget_usd,
        )

    def create_handoff(
        self,
        new_receiver: str,
        handoff_metrics: HandoffMetrics,
        *,
        updated_payload: dict[str, PayloadValue] | None = None,
        cost_budget_remaining_usd: float | None = None,
    ) -> "MessageEnvelope":
        """Create a HANDOFF envelope that transfers this task to another agent.

        Parameters
        ----------
        new_receiver:
            Agent ID that should take over this task.
        handoff_metrics:
            Metrics captured at the handoff point (cost, tokens, latency).
        updated_payload:
            Optional replacement payload.  If None, the original payload is
            carried forward unchanged.
        cost_budget_remaining_usd:
            Remaining cost budget after accounting for work already done.
            If None, the original ``cost_budget_usd`` is preserved.

        Returns
        -------
        MessageEnvelope
            A new HANDOFF envelope with updated routing and embedded metrics.
        """
        metrics_dict: dict[str, PayloadValue] = {
            "source_agent": handoff_metrics.source_agent,
            "target_agent": handoff_metrics.target_agent,
            "reason": handoff_metrics.reason,
            "cost_incurred_usd": handoff_metrics.cost_incurred_usd,
            "tokens_consumed": handoff_metrics.tokens_consumed,
            "latency_ms": handoff_metrics.latency_ms,
            "handoff_timestamp": handoff_metrics.handoff_timestamp,
            "attempt_number": handoff_metrics.attempt_number,
        }
        merged_payload: dict[str, PayloadValue] = {
            **(updated_payload if updated_payload is not None else self.payload),
            "__handoff_metrics__": metrics_dict,  # type: ignore[dict-item]
        }
        return MessageEnvelope(
            sender=self.receiver,
            receiver=new_receiver,
            payload=merged_payload,
            message_type=MessageType.HANDOFF,
            priority=self.priority,
            topic=self.topic,
            trace_id=self.trace_id,
            parent_message_id=self.message_id,
            cost_budget_usd=(
                cost_budget_remaining_usd
                if cost_budget_remaining_usd is not None
                else self.cost_budget_usd
            ),
            metadata=dict(self.metadata),
        )

    def create_ack(self) -> "MessageEnvelope":
        """Create an ACK envelope acknowledging receipt of this message.

        Returns
        -------
        MessageEnvelope
            A minimal ACK envelope carrying only the original message_id.
        """
        return self.create_response(
            payload={"acked_message_id": self.message_id},
            message_type=MessageType.ACK,
            priority=Priority.HIGH,
        )

    def create_error(
        self,
        error_message: str,
        *,
        error_code: str = "PROCESSING_ERROR",
    ) -> "MessageEnvelope":
        """Create an ERROR envelope signalling a processing failure.

        Parameters
        ----------
        error_message:
            Human-readable description of the failure.
        error_code:
            Machine-readable error code for programmatic handling.

        Returns
        -------
        MessageEnvelope
            An ERROR envelope addressed back to the original sender.
        """
        return self.create_response(
            payload={
                "error_code": error_code,
                "error_message": error_message,
                "failed_message_id": self.message_id,
            },
            message_type=MessageType.ERROR,
            priority=Priority.HIGH,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_json(self) -> bytes:
        """Serialize this envelope to UTF-8 encoded JSON bytes.

        The ``message_type`` and ``priority`` enum values are serialized
        to their Python names so they survive round-trips correctly.

        Returns
        -------
        bytes
            UTF-8 JSON encoding of this envelope.
        """
        data = {
            "sender": self.sender,
            "receiver": self.receiver,
            "payload": self.payload,
            "message_type": self.message_type.value,
            "priority": int(self.priority),
            "topic": self.topic,
            "trace_id": self.trace_id,
            "parent_message_id": self.parent_message_id,
            "message_id": self.message_id,
            "created_at": self.created_at,
            "ttl_seconds": self.ttl_seconds,
            "cost_budget_usd": self.cost_budget_usd,
            "metadata": self.metadata,
        }
        return json.dumps(data, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_json(cls, raw: bytes | str) -> "MessageEnvelope":
        """Deserialize an envelope from UTF-8 JSON bytes or a JSON string.

        Parameters
        ----------
        raw:
            JSON bytes or string previously produced by ``to_json()``.

        Returns
        -------
        MessageEnvelope
            Reconstructed envelope.

        Raises
        ------
        ValueError
            If the JSON is malformed or required fields are missing.
        """
        try:
            data: dict[str, PayloadValue] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse MessageEnvelope JSON: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError(
                f"MessageEnvelope JSON must be an object, got {type(data).__name__}."
            )

        required_fields = ("sender", "receiver", "payload", "message_id", "trace_id")
        for required_field in required_fields:
            if required_field not in data:
                raise ValueError(
                    f"MessageEnvelope JSON is missing required field: {required_field!r}."
                )

        try:
            message_type = MessageType(data["message_type"])
        except (ValueError, KeyError) as exc:
            raise ValueError(f"Invalid message_type in envelope JSON: {exc}") from exc

        try:
            priority = Priority(int(data["priority"]))  # type: ignore[arg-type]
        except (ValueError, KeyError) as exc:
            raise ValueError(f"Invalid priority in envelope JSON: {exc}") from exc

        return cls(
            sender=str(data["sender"]),
            receiver=str(data["receiver"]),
            payload=dict(data["payload"]),  # type: ignore[arg-type]
            message_type=message_type,
            priority=priority,
            topic=data.get("topic"),  # type: ignore[arg-type]
            trace_id=str(data["trace_id"]),
            parent_message_id=(
                str(data["parent_message_id"])
                if data.get("parent_message_id") is not None
                else None
            ),
            message_id=str(data["message_id"]),
            created_at=float(data.get("created_at", time.time())),  # type: ignore[arg-type]
            ttl_seconds=(
                float(data["ttl_seconds"])  # type: ignore[arg-type]
                if data.get("ttl_seconds") is not None
                else None
            ),
            cost_budget_usd=(
                float(data["cost_budget_usd"])  # type: ignore[arg-type]
                if data.get("cost_budget_usd") is not None
                else None
            ),
            metadata=dict(data.get("metadata", {})),  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, PayloadValue]:
        """Return a JSON-serializable dict representation of this envelope.

        The returned dict is the same structure produced by ``to_json()``
        before the final JSON encoding step.

        Returns
        -------
        dict
            Plain Python dict ready for ``json.dumps``.
        """
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "payload": self.payload,
            "message_type": self.message_type.value,
            "priority": int(self.priority),
            "topic": self.topic,
            "trace_id": self.trace_id,
            "parent_message_id": self.parent_message_id,
            "message_id": self.message_id,
            "created_at": self.created_at,
            "ttl_seconds": self.ttl_seconds,
            "cost_budget_usd": self.cost_budget_usd,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"MessageEnvelope("
            f"id={self.message_id[:8]}…, "
            f"type={self.message_type.value}, "
            f"priority={self.priority.name}, "
            f"sender={self.sender!r}, "
            f"receiver={self.receiver!r}"
            f")"
        )
