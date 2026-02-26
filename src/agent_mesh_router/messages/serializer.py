"""Message serialization — JSON and optional MessagePack backends.

``MessageSerializer`` is a stateless utility class with class methods for
encoding and decoding ``MessageEnvelope`` objects.

Two formats are supported:

* **json** — always available; produces UTF-8 encoded JSON bytes.
* **msgpack** — available when the ``msgpack`` package is installed;
  produces compact binary frames.  Falls back gracefully to JSON when
  msgpack is not installed.

Usage
-----
::

    from agent_mesh_router.messages.serializer import MessageSerializer
    from agent_mesh_router.messages.envelope import MessageEnvelope
    from agent_mesh_router.messages.types import MessageType, Priority

    env = MessageEnvelope(
        sender="a",
        receiver="b",
        payload={"cmd": "run"},
    )

    # JSON (default)
    raw = MessageSerializer.encode(env)
    env2 = MessageSerializer.decode(raw)

    # MessagePack (if installed)
    raw_mp = MessageSerializer.encode(env, format="msgpack")
    env3 = MessageSerializer.decode(raw_mp, format="msgpack")
"""
from __future__ import annotations

from typing import Literal

from agent_mesh_router.messages.envelope import MessageEnvelope
from agent_mesh_router.messages.types import MessageType, Priority

SerializationFormat = Literal["json", "msgpack"]

_MSGPACK_HEADER = b"\xc1"  # Reserved byte used as a simple format sentinel.


class MessageSerializer:
    """Stateless serializer for ``MessageEnvelope`` objects.

    All methods are class methods — no instantiation needed.
    """

    @classmethod
    def encode(
        cls,
        envelope: MessageEnvelope,
        *,
        format: SerializationFormat = "json",  # noqa: A002
    ) -> bytes:
        """Encode a ``MessageEnvelope`` to bytes.

        Parameters
        ----------
        envelope:
            The envelope to encode.
        format:
            ``"json"`` (default) or ``"msgpack"``.  If ``"msgpack"`` is
            requested but the package is unavailable, falls back to JSON
            transparently and prefixes the output with a sentinel byte so
            ``decode`` can detect the fallback automatically.

        Returns
        -------
        bytes
            Encoded bytes ready for transmission or storage.
        """
        if format == "msgpack":
            return cls._encode_msgpack(envelope)
        return envelope.to_json()

    @classmethod
    def decode(
        cls,
        raw: bytes,
        *,
        format: SerializationFormat = "json",  # noqa: A002
    ) -> MessageEnvelope:
        """Decode bytes back into a ``MessageEnvelope``.

        Parameters
        ----------
        raw:
            Bytes previously produced by ``encode``.
        format:
            Must match the format used during encoding.  When ``"msgpack"``
            is specified and the msgpack library is unavailable, the JSON
            fallback path is attempted automatically.

        Returns
        -------
        MessageEnvelope
            Reconstructed envelope.

        Raises
        ------
        ValueError
            If decoding fails or the bytes are malformed.
        """
        if format == "msgpack":
            return cls._decode_msgpack(raw)
        return MessageEnvelope.from_json(raw)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _envelope_to_wire_dict(cls, envelope: MessageEnvelope) -> dict[str, object]:
        """Convert an envelope to a wire-format dict safe for msgpack."""
        return {
            "sender": envelope.sender,
            "receiver": envelope.receiver,
            "payload": envelope.payload,
            "message_type": envelope.message_type.value,
            "priority": int(envelope.priority),
            "topic": envelope.topic,
            "trace_id": envelope.trace_id,
            "parent_message_id": envelope.parent_message_id,
            "message_id": envelope.message_id,
            "created_at": envelope.created_at,
            "ttl_seconds": envelope.ttl_seconds,
            "cost_budget_usd": envelope.cost_budget_usd,
            "metadata": envelope.metadata,
        }

    @classmethod
    def _wire_dict_to_envelope(cls, data: dict[str, object]) -> MessageEnvelope:
        """Reconstruct an envelope from a wire-format dict."""
        return MessageEnvelope(
            sender=str(data["sender"]),
            receiver=str(data["receiver"]),
            payload=dict(data["payload"]),  # type: ignore[arg-type]
            message_type=MessageType(data["message_type"]),
            priority=Priority(int(data["priority"])),  # type: ignore[arg-type]
            topic=data.get("topic"),  # type: ignore[arg-type]
            trace_id=str(data["trace_id"]),
            parent_message_id=(
                str(data["parent_message_id"])
                if data.get("parent_message_id") is not None
                else None
            ),
            message_id=str(data["message_id"]),
            created_at=float(data.get("created_at", 0.0)),  # type: ignore[arg-type]
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

    @classmethod
    def _encode_msgpack(cls, envelope: MessageEnvelope) -> bytes:
        """Encode using msgpack, falling back to JSON if unavailable."""
        try:
            import msgpack  # type: ignore[import]

            wire = cls._envelope_to_wire_dict(envelope)
            packed: bytes = msgpack.packb(wire, use_bin_type=True)
            return packed
        except ImportError:
            # Fall back to JSON with a sentinel prefix so decode knows.
            return _MSGPACK_HEADER + envelope.to_json()

    @classmethod
    def _decode_msgpack(cls, raw: bytes) -> MessageEnvelope:
        """Decode msgpack bytes, handling the JSON fallback sentinel."""
        if raw.startswith(_MSGPACK_HEADER):
            # JSON fallback path.
            return MessageEnvelope.from_json(raw[len(_MSGPACK_HEADER):])

        try:
            import msgpack  # type: ignore[import]

            data: dict[str, object] = msgpack.unpackb(raw, raw=False)
            return cls._wire_dict_to_envelope(data)
        except ImportError as exc:
            raise ValueError(
                "msgpack is not installed. Install it with: pip install msgpack"
            ) from exc
        except Exception as exc:
            raise ValueError(f"Failed to decode msgpack envelope: {exc}") from exc

    @classmethod
    def supports_msgpack(cls) -> bool:
        """Return True if the msgpack library is currently installed.

        Returns
        -------
        bool
            True when msgpack encoding/decoding is available natively.
        """
        try:
            import msgpack  # noqa: F401  # type: ignore[import]

            return True
        except ImportError:
            return False
