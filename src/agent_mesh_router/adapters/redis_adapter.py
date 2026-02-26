"""RedisTransport — deliver MessageEnvelopes via Redis pub/sub.

Envelopes are serialized to JSON and published to a Redis channel derived
from the envelope's ``receiver`` field (``amr:<receiver>``).  Subscribers
can listen on that channel to receive messages.

Optional dependency
-------------------
Requires ``redis[asyncio]``.  Import is guarded so the rest of the package
remains importable when Redis is not installed.

Install with::

    pip install agent-mesh-router[redis]

Example
-------
::

    import asyncio
    from agent_mesh_router.adapters.redis_adapter import RedisTransport
    from agent_mesh_router.messages.envelope import MessageEnvelope

    transport = RedisTransport(url="redis://localhost:6379/0")
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
    import redis.asyncio as aioredis  # type: ignore[import-untyped]

    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False

#: Channel name prefix to namespace agent channels in Redis.
CHANNEL_PREFIX: str = "amr"


class RedisTransportError(RuntimeError):
    """Raised when a Redis pub/sub delivery fails."""


class RedisTransport:
    """Publish ``MessageEnvelope`` objects to Redis pub/sub channels.

    Each envelope is published to the channel ``amr:<receiver>`` (or
    ``amr:<topic>`` when the envelope has a topic set).

    Parameters
    ----------
    url:
        Redis connection URL (e.g. ``"redis://localhost:6379/0"``).
    channel_prefix:
        Prefix for all agent channels.  Defaults to ``"amr"``.

    Raises
    ------
    ImportError
        At construction time if ``redis[asyncio]`` is not installed.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        *,
        channel_prefix: str = CHANNEL_PREFIX,
    ) -> None:
        if not _REDIS_AVAILABLE:
            raise ImportError(
                "RedisTransport requires 'redis[asyncio]'. "
                "Install it with: pip install agent-mesh-router[redis]"
            )
        self._url = url
        self._channel_prefix = channel_prefix
        self._client: aioredis.Redis | None = None  # type: ignore[name-defined]
        self._total_published: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Establish a connection to the Redis server."""
        self._client = aioredis.from_url(  # type: ignore[union-attr]
            self._url, encoding="utf-8", decode_responses=False
        )
        await self._client.ping()
        logger.info("RedisTransport: connected to %r.", self._url)

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        logger.info("RedisTransport: connection closed.")

    # ------------------------------------------------------------------
    # Send
    # ------------------------------------------------------------------

    async def send(self, envelope: MessageEnvelope) -> int:
        """Publish an envelope to the appropriate Redis channel.

        Parameters
        ----------
        envelope:
            Envelope to publish.

        Returns
        -------
        int
            Number of subscribers that received the message (Redis PUBLISH
            return value).

        Raises
        ------
        RedisTransportError
            If the client is not connected or the PUBLISH command fails.
        """
        if self._client is None:
            raise RedisTransportError(
                "RedisTransport is not connected. Call connect() first."
            )

        channel = self._build_channel(envelope)
        payload = envelope.to_json()

        try:
            subscriber_count: int = await self._client.publish(channel, payload)
            self._total_published += 1
            logger.debug(
                "RedisTransport: published message %s to channel %r "
                "(%d subscriber(s)).",
                envelope.message_id[:8],
                channel,
                subscriber_count,
            )
            return subscriber_count
        except Exception as exc:
            raise RedisTransportError(
                f"Failed to publish message {envelope.message_id[:8]} "
                f"to channel {channel!r}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Subscribe helper
    # ------------------------------------------------------------------

    async def subscribe_channel(
        self, agent_id: str
    ) -> object:
        """Return a Redis pubsub object subscribed to the agent's channel.

        The caller is responsible for listening on the returned pubsub
        object and calling ``close()`` when done.

        Parameters
        ----------
        agent_id:
            Agent whose channel to subscribe to.

        Returns
        -------
        redis.asyncio.client.PubSub
            An active pub/sub object.

        Raises
        ------
        RedisTransportError
            If not connected.
        """
        if self._client is None:
            raise RedisTransportError(
                "RedisTransport is not connected. Call connect() first."
            )
        channel = f"{self._channel_prefix}:{agent_id}"
        pubsub = self._client.pubsub()
        await pubsub.subscribe(channel)
        return pubsub

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def url(self) -> str:
        """Configured Redis connection URL."""
        return self._url

    @property
    def total_published(self) -> int:
        """Total envelopes successfully published."""
        return self._total_published

    @property
    def is_connected(self) -> bool:
        """Return True if a Redis client is open."""
        return self._client is not None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_channel(self, envelope: MessageEnvelope) -> str:
        """Derive the Redis channel name from the envelope."""
        if envelope.topic:
            return f"{self._channel_prefix}:topic:{envelope.topic}"
        return f"{self._channel_prefix}:{envelope.receiver}"

    def __repr__(self) -> str:
        return (
            f"RedisTransport("
            f"url={self._url!r}, "
            f"connected={self.is_connected}, "
            f"published={self._total_published}"
            f")"
        )
