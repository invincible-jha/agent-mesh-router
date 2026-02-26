"""KafkaTransport — deliver MessageEnvelopes via Apache Kafka.

Envelopes are serialized to JSON bytes and produced to a Kafka topic.
The Kafka topic is derived from the envelope's ``topic`` field, falling
back to a configurable ``default_topic``.

Optional dependency
-------------------
Requires ``aiokafka``.  Import is guarded so the rest of the package
remains importable when Kafka is not installed.

Install with::

    pip install agent-mesh-router[kafka]

Example
-------
::

    import asyncio
    from agent_mesh_router.adapters.kafka_adapter import KafkaTransport
    from agent_mesh_router.messages.envelope import MessageEnvelope

    transport = KafkaTransport(
        bootstrap_servers="localhost:9092",
        default_topic="agent-mesh-router",
    )
    await transport.connect()

    envelope = MessageEnvelope(
        sender="agent-a", receiver="agent-b", payload={"task": "run"}
    )
    await transport.send(envelope)
    await transport.close()
"""
from __future__ import annotations

import logging

from agent_mesh_router.messages.envelope import MessageEnvelope

logger = logging.getLogger(__name__)

try:
    from aiokafka import AIOKafkaProducer  # type: ignore[import-untyped]

    _KAFKA_AVAILABLE = True
except ImportError:
    _KAFKA_AVAILABLE = False


class KafkaTransportError(RuntimeError):
    """Raised when a Kafka delivery fails."""


class KafkaTransport:
    """Produce ``MessageEnvelope`` objects to a Kafka topic.

    Parameters
    ----------
    bootstrap_servers:
        Comma-separated list of Kafka broker addresses.
    default_topic:
        Default Kafka topic to produce to when ``envelope.topic`` is None.
    key_by_receiver:
        When True (default), the Kafka message key is set to
        ``envelope.receiver.encode()`` so messages for the same agent are
        partitioned together.

    Raises
    ------
    ImportError
        At construction time if ``aiokafka`` is not installed.
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        *,
        default_topic: str = "agent-mesh-router",
        key_by_receiver: bool = True,
    ) -> None:
        if not _KAFKA_AVAILABLE:
            raise ImportError(
                "KafkaTransport requires 'aiokafka'. "
                "Install it with: pip install agent-mesh-router[kafka]"
            )
        self._bootstrap_servers = bootstrap_servers
        self._default_topic = default_topic
        self._key_by_receiver = key_by_receiver
        self._producer: AIOKafkaProducer | None = None  # type: ignore[name-defined]
        self._total_produced: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Start the Kafka producer and connect to the cluster."""
        self._producer = AIOKafkaProducer(  # type: ignore[name-defined]
            bootstrap_servers=self._bootstrap_servers,
            value_serializer=lambda v: v,  # We pass pre-serialized bytes
        )
        await self._producer.start()
        logger.info(
            "KafkaTransport: connected to %r.", self._bootstrap_servers
        )

    async def close(self) -> None:
        """Flush pending messages and stop the producer."""
        if self._producer is not None:
            await self._producer.flush()
            await self._producer.stop()
            self._producer = None
        logger.info("KafkaTransport: producer stopped.")

    # ------------------------------------------------------------------
    # Send
    # ------------------------------------------------------------------

    async def send(
        self,
        envelope: MessageEnvelope,
        *,
        topic: str | None = None,
    ) -> None:
        """Produce an envelope to a Kafka topic.

        Parameters
        ----------
        envelope:
            Envelope to produce.
        topic:
            Override topic.  If None, uses ``envelope.topic`` or
            ``default_topic`` in that order.

        Raises
        ------
        KafkaTransportError
            If the producer is not started or the send fails.
        """
        if self._producer is None:
            raise KafkaTransportError(
                "KafkaTransport is not connected. Call connect() first."
            )

        target_topic = topic or envelope.topic or self._default_topic
        payload_bytes = envelope.to_json()
        key: bytes | None = (
            envelope.receiver.encode("utf-8") if self._key_by_receiver else None
        )

        try:
            await self._producer.send_and_wait(
                target_topic,
                value=payload_bytes,
                key=key,
                headers=[
                    ("x-trace-id", envelope.trace_id.encode()),
                    ("x-message-id", envelope.message_id.encode()),
                    ("x-message-type", envelope.message_type.value.encode()),
                ],
            )
            self._total_produced += 1
            logger.debug(
                "KafkaTransport: produced message %s to topic %r.",
                envelope.message_id[:8],
                target_topic,
            )
        except Exception as exc:
            raise KafkaTransportError(
                f"Failed to produce message {envelope.message_id[:8]} "
                f"to topic {target_topic!r}: {exc}"
            ) from exc

    async def send_many(self, envelopes: list[MessageEnvelope]) -> None:
        """Produce multiple envelopes in a batch.

        Each envelope is produced independently but without waiting for
        individual acknowledgements.  A final ``flush()`` ensures delivery.

        Parameters
        ----------
        envelopes:
            Envelopes to produce.

        Raises
        ------
        KafkaTransportError
            If the producer is not started.
        """
        if self._producer is None:
            raise KafkaTransportError(
                "KafkaTransport is not connected. Call connect() first."
            )
        for envelope in envelopes:
            target_topic = envelope.topic or self._default_topic
            payload_bytes = envelope.to_json()
            key = (
                envelope.receiver.encode("utf-8") if self._key_by_receiver else None
            )
            await self._producer.send(
                target_topic,
                value=payload_bytes,
                key=key,
            )
            self._total_produced += 1

        await self._producer.flush()
        logger.debug(
            "KafkaTransport: flushed batch of %d messages.", len(envelopes)
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def bootstrap_servers(self) -> str:
        """Configured Kafka bootstrap servers string."""
        return self._bootstrap_servers

    @property
    def default_topic(self) -> str:
        """Default Kafka topic for envelopes without an explicit topic."""
        return self._default_topic

    @property
    def total_produced(self) -> int:
        """Total envelopes successfully produced."""
        return self._total_produced

    @property
    def is_connected(self) -> bool:
        """Return True if a Kafka producer is active."""
        return self._producer is not None

    def __repr__(self) -> str:
        return (
            f"KafkaTransport("
            f"servers={self._bootstrap_servers!r}, "
            f"topic={self._default_topic!r}, "
            f"connected={self.is_connected}, "
            f"produced={self._total_produced}"
            f")"
        )
