"""Tests for HTTP, Kafka, and Redis transport adapters (mocked clients)."""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_mesh_router.messages.envelope import MessageEnvelope


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _envelope(receiver: str = "agent-b") -> MessageEnvelope:
    return MessageEnvelope(
        sender="agent-a",
        receiver=receiver,
        payload={"task": "run"},
    )


# ===========================================================================
# HttpTransport
# ===========================================================================


class TestHttpTransport:
    """HttpTransport tests using a mocked httpx.AsyncClient."""

    def _make_response(self, status_code: int, body: dict | None = None) -> MagicMock:
        response = MagicMock()
        response.status_code = status_code
        response.is_success = 200 <= status_code < 300
        if body is not None:
            response.json.return_value = body
        else:
            response.json.side_effect = ValueError("no json")
        return response

    @pytest.mark.asyncio
    async def test_send_success(self) -> None:
        from agent_mesh_router.adapters.http import HttpTransport

        response = self._make_response(200, {"ok": True})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=response)

        with patch("httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            transport = HttpTransport("http://localhost:8080", max_retries=0)
            result = await transport.send(_envelope())

        assert result == {"ok": True}
        assert transport.total_sent == 1
        assert transport.total_failed == 0

    @pytest.mark.asyncio
    async def test_send_success_non_json_response(self) -> None:
        from agent_mesh_router.adapters.http import HttpTransport

        response = self._make_response(200)
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=response)

        with patch("httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            transport = HttpTransport("http://localhost:8080", max_retries=0)
            result = await transport.send(_envelope())

        assert result == {"status_code": 200}

    @pytest.mark.asyncio
    async def test_send_4xx_permanent_failure(self) -> None:
        from agent_mesh_router.adapters.http import HttpTransport, HttpTransportError

        response = self._make_response(400)
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=response)

        with patch("httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            transport = HttpTransport("http://localhost:8080", max_retries=2)
            with pytest.raises(HttpTransportError):
                await transport.send(_envelope())

        assert transport.total_failed == 1
        # Should not retry on 4xx.
        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_send_5xx_retries_and_fails(self) -> None:
        from agent_mesh_router.adapters.http import HttpTransport, HttpTransportError

        response = self._make_response(503)
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=response)

        with patch("httpx.AsyncClient") as mock_cls, patch("asyncio.sleep"):
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            transport = HttpTransport(
                "http://localhost:8080", max_retries=2, backoff_base_seconds=0.0
            )
            with pytest.raises(HttpTransportError):
                await transport.send(_envelope())

        assert transport.total_failed == 1
        assert mock_client.post.call_count == 3  # 1 initial + 2 retries

    @pytest.mark.asyncio
    async def test_send_network_error_retries(self) -> None:
        from agent_mesh_router.adapters.http import HttpTransport, HttpTransportError

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=ConnectionError("refused"))

        with patch("httpx.AsyncClient") as mock_cls, patch("asyncio.sleep"):
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            transport = HttpTransport(
                "http://localhost:8080", max_retries=1, backoff_base_seconds=0.0
            )
            with pytest.raises(HttpTransportError):
                await transport.send(_envelope())

        assert transport.total_failed == 1

    @pytest.mark.asyncio
    async def test_send_many_returns_list(self) -> None:
        from agent_mesh_router.adapters.http import HttpTransport

        response = self._make_response(200, {"ok": True})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=response)

        with patch("httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            transport = HttpTransport("http://localhost:8080", max_retries=0)
            results = await transport.send_many([_envelope(), _envelope()])

        assert len(results) == 2
        assert transport.total_sent == 2

    @pytest.mark.asyncio
    async def test_send_many_individual_failure_captured(self) -> None:
        from agent_mesh_router.adapters.http import HttpTransport

        response = self._make_response(400)
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=response)

        with patch("httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            transport = HttpTransport("http://localhost:8080", max_retries=0)
            results = await transport.send_many([_envelope()])

        assert "error" in results[0]

    def test_invalid_max_retries_raises(self) -> None:
        from agent_mesh_router.adapters.http import HttpTransport

        with pytest.raises(ValueError):
            HttpTransport("http://localhost", max_retries=-1)

    def test_invalid_timeout_raises(self) -> None:
        from agent_mesh_router.adapters.http import HttpTransport

        with pytest.raises(ValueError):
            HttpTransport("http://localhost", timeout_seconds=0.0)

    def test_properties(self) -> None:
        from agent_mesh_router.adapters.http import HttpTransport

        transport = HttpTransport("http://example.com/", max_retries=0)
        assert transport.base_url == "http://example.com"
        assert transport.total_sent == 0
        assert transport.total_failed == 0

    def test_repr(self) -> None:
        from agent_mesh_router.adapters.http import HttpTransport

        transport = HttpTransport("http://example.com", max_retries=0)
        assert "example.com" in repr(transport)


# ===========================================================================
# KafkaTransport
# ===========================================================================


class TestKafkaTransport:
    """KafkaTransport tests with aiokafka patched into sys.modules."""

    def _make_producer(self) -> AsyncMock:
        producer = AsyncMock()
        producer.start = AsyncMock()
        producer.stop = AsyncMock()
        producer.flush = AsyncMock()
        producer.send_and_wait = AsyncMock()
        producer.send = AsyncMock()
        return producer

    def _patch_kafka(self, producer: AsyncMock):
        """Return a context manager that patches aiokafka."""
        mock_aiokafka = MagicMock()
        mock_aiokafka.AIOKafkaProducer = MagicMock(return_value=producer)
        return patch.dict("sys.modules", {"aiokafka": mock_aiokafka})

    @pytest.mark.asyncio
    async def test_connect_starts_producer(self) -> None:
        producer = self._make_producer()
        with self._patch_kafka(producer):
            # Force re-import to pick up mock.
            import importlib
            import agent_mesh_router.adapters.kafka_adapter as km
            km._KAFKA_AVAILABLE = True
            km.AIOKafkaProducer = MagicMock(return_value=producer)

            from agent_mesh_router.adapters.kafka_adapter import KafkaTransport
            transport = KafkaTransport.__new__(KafkaTransport)
            transport._bootstrap_servers = "localhost:9092"
            transport._default_topic = "test-topic"
            transport._key_by_receiver = True
            transport._producer = None
            transport._total_produced = 0
            transport._producer = producer
            await producer.start()

        producer.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_not_connected_raises(self) -> None:
        from agent_mesh_router.adapters.kafka_adapter import (
            KafkaTransport,
            KafkaTransportError,
        )
        import agent_mesh_router.adapters.kafka_adapter as km

        # Simulate Kafka being available.
        km._KAFKA_AVAILABLE = True
        km.AIOKafkaProducer = MagicMock(return_value=self._make_producer())

        transport = KafkaTransport.__new__(KafkaTransport)
        transport._bootstrap_servers = "localhost:9092"
        transport._default_topic = "test"
        transport._key_by_receiver = True
        transport._producer = None
        transport._total_produced = 0

        with pytest.raises(KafkaTransportError, match="not connected"):
            await transport.send(_envelope())

    @pytest.mark.asyncio
    async def test_send_publishes_to_topic(self) -> None:
        producer = self._make_producer()

        from agent_mesh_router.adapters.kafka_adapter import (
            KafkaTransport,
        )
        import agent_mesh_router.adapters.kafka_adapter as km

        km._KAFKA_AVAILABLE = True
        km.AIOKafkaProducer = MagicMock(return_value=producer)

        transport = KafkaTransport.__new__(KafkaTransport)
        transport._bootstrap_servers = "localhost:9092"
        transport._default_topic = "my-topic"
        transport._key_by_receiver = True
        transport._producer = producer
        transport._total_produced = 0

        await transport.send(_envelope())
        producer.send_and_wait.assert_called_once()
        assert transport.total_produced == 1

    @pytest.mark.asyncio
    async def test_send_producer_error_raises(self) -> None:
        producer = self._make_producer()
        producer.send_and_wait.side_effect = RuntimeError("kafka down")

        from agent_mesh_router.adapters.kafka_adapter import (
            KafkaTransport,
            KafkaTransportError,
        )
        import agent_mesh_router.adapters.kafka_adapter as km

        km._KAFKA_AVAILABLE = True
        transport = KafkaTransport.__new__(KafkaTransport)
        transport._bootstrap_servers = "localhost:9092"
        transport._default_topic = "my-topic"
        transport._key_by_receiver = True
        transport._producer = producer
        transport._total_produced = 0

        with pytest.raises(KafkaTransportError):
            await transport.send(_envelope())

    @pytest.mark.asyncio
    async def test_send_many_flushes(self) -> None:
        producer = self._make_producer()

        from agent_mesh_router.adapters.kafka_adapter import KafkaTransport
        import agent_mesh_router.adapters.kafka_adapter as km

        km._KAFKA_AVAILABLE = True
        transport = KafkaTransport.__new__(KafkaTransport)
        transport._bootstrap_servers = "localhost:9092"
        transport._default_topic = "my-topic"
        transport._key_by_receiver = True
        transport._producer = producer
        transport._total_produced = 0

        await transport.send_many([_envelope(), _envelope()])
        producer.flush.assert_called_once()
        assert transport.total_produced == 2

    @pytest.mark.asyncio
    async def test_send_many_not_connected_raises(self) -> None:
        from agent_mesh_router.adapters.kafka_adapter import (
            KafkaTransport,
            KafkaTransportError,
        )
        import agent_mesh_router.adapters.kafka_adapter as km

        km._KAFKA_AVAILABLE = True
        transport = KafkaTransport.__new__(KafkaTransport)
        transport._bootstrap_servers = "localhost:9092"
        transport._default_topic = "my-topic"
        transport._key_by_receiver = True
        transport._producer = None
        transport._total_produced = 0

        with pytest.raises(KafkaTransportError):
            await transport.send_many([_envelope()])

    @pytest.mark.asyncio
    async def test_close_flushes_and_stops(self) -> None:
        producer = self._make_producer()

        from agent_mesh_router.adapters.kafka_adapter import KafkaTransport
        import agent_mesh_router.adapters.kafka_adapter as km

        km._KAFKA_AVAILABLE = True
        transport = KafkaTransport.__new__(KafkaTransport)
        transport._bootstrap_servers = "localhost:9092"
        transport._default_topic = "my-topic"
        transport._key_by_receiver = False
        transport._producer = producer
        transport._total_produced = 0

        await transport.close()
        producer.flush.assert_called_once()
        producer.stop.assert_called_once()
        assert transport._producer is None

    def test_properties(self) -> None:
        from agent_mesh_router.adapters.kafka_adapter import KafkaTransport
        import agent_mesh_router.adapters.kafka_adapter as km

        km._KAFKA_AVAILABLE = True
        producer = self._make_producer()
        transport = KafkaTransport.__new__(KafkaTransport)
        transport._bootstrap_servers = "localhost:9092"
        transport._default_topic = "topic-x"
        transport._key_by_receiver = True
        transport._producer = None
        transport._total_produced = 5

        assert transport.bootstrap_servers == "localhost:9092"
        assert transport.default_topic == "topic-x"
        assert transport.total_produced == 5
        assert transport.is_connected is False

    def test_repr(self) -> None:
        from agent_mesh_router.adapters.kafka_adapter import KafkaTransport
        import agent_mesh_router.adapters.kafka_adapter as km

        km._KAFKA_AVAILABLE = True
        transport = KafkaTransport.__new__(KafkaTransport)
        transport._bootstrap_servers = "broker:9092"
        transport._default_topic = "t"
        transport._key_by_receiver = True
        transport._producer = None
        transport._total_produced = 0

        assert "broker:9092" in repr(transport)


# ===========================================================================
# RedisTransport
# ===========================================================================


class TestRedisTransport:
    """RedisTransport tests with redis.asyncio patched."""

    def _make_redis_client(self) -> AsyncMock:
        client = AsyncMock()
        client.ping = AsyncMock()
        client.publish = AsyncMock(return_value=1)
        client.aclose = AsyncMock()
        pubsub = AsyncMock()
        pubsub.subscribe = AsyncMock()
        client.pubsub = MagicMock(return_value=pubsub)
        return client

    @pytest.mark.asyncio
    async def test_connect_pings_redis(self) -> None:
        redis_client = self._make_redis_client()

        from agent_mesh_router.adapters.redis_adapter import RedisTransport
        import agent_mesh_router.adapters.redis_adapter as rm

        rm._REDIS_AVAILABLE = True

        transport = RedisTransport.__new__(RedisTransport)
        transport._url = "redis://localhost:6379"
        transport._channel_prefix = "amr"
        transport._client = None
        transport._total_published = 0

        # Simulate connect() calling aioredis.from_url.
        with patch(
            "agent_mesh_router.adapters.redis_adapter.aioredis",
            create=True,
        ) as mock_aioredis:
            mock_aioredis.from_url = MagicMock(return_value=redis_client)
            rm._REDIS_AVAILABLE = True
            transport._client = redis_client
            await redis_client.ping()

        redis_client.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_not_connected_raises(self) -> None:
        from agent_mesh_router.adapters.redis_adapter import (
            RedisTransport,
            RedisTransportError,
        )
        import agent_mesh_router.adapters.redis_adapter as rm

        rm._REDIS_AVAILABLE = True
        transport = RedisTransport.__new__(RedisTransport)
        transport._url = "redis://localhost"
        transport._channel_prefix = "amr"
        transport._client = None
        transport._total_published = 0

        with pytest.raises(RedisTransportError, match="not connected"):
            await transport.send(_envelope())

    @pytest.mark.asyncio
    async def test_send_publishes_and_counts(self) -> None:
        redis_client = self._make_redis_client()

        from agent_mesh_router.adapters.redis_adapter import RedisTransport
        import agent_mesh_router.adapters.redis_adapter as rm

        rm._REDIS_AVAILABLE = True
        transport = RedisTransport.__new__(RedisTransport)
        transport._url = "redis://localhost"
        transport._channel_prefix = "amr"
        transport._client = redis_client
        transport._total_published = 0

        count = await transport.send(_envelope())
        assert count == 1
        assert transport.total_published == 1
        redis_client.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_uses_topic_channel_when_set(self) -> None:
        redis_client = self._make_redis_client()

        from agent_mesh_router.adapters.redis_adapter import RedisTransport
        import agent_mesh_router.adapters.redis_adapter as rm

        rm._REDIS_AVAILABLE = True
        transport = RedisTransport.__new__(RedisTransport)
        transport._url = "redis://localhost"
        transport._channel_prefix = "amr"
        transport._client = redis_client
        transport._total_published = 0

        env = _envelope()
        env.topic = "my-topic"
        await transport.send(env)
        channel_arg = redis_client.publish.call_args[0][0]
        assert "topic" in channel_arg
        assert "my-topic" in channel_arg

    @pytest.mark.asyncio
    async def test_send_publish_error_raises(self) -> None:
        redis_client = self._make_redis_client()
        redis_client.publish.side_effect = RuntimeError("redis error")

        from agent_mesh_router.adapters.redis_adapter import (
            RedisTransport,
            RedisTransportError,
        )
        import agent_mesh_router.adapters.redis_adapter as rm

        rm._REDIS_AVAILABLE = True
        transport = RedisTransport.__new__(RedisTransport)
        transport._url = "redis://localhost"
        transport._channel_prefix = "amr"
        transport._client = redis_client
        transport._total_published = 0

        with pytest.raises(RedisTransportError):
            await transport.send(_envelope())

    @pytest.mark.asyncio
    async def test_subscribe_channel_returns_pubsub(self) -> None:
        redis_client = self._make_redis_client()

        from agent_mesh_router.adapters.redis_adapter import RedisTransport
        import agent_mesh_router.adapters.redis_adapter as rm

        rm._REDIS_AVAILABLE = True
        transport = RedisTransport.__new__(RedisTransport)
        transport._url = "redis://localhost"
        transport._channel_prefix = "amr"
        transport._client = redis_client
        transport._total_published = 0

        pubsub = await transport.subscribe_channel("agent-x")
        assert pubsub is not None

    @pytest.mark.asyncio
    async def test_subscribe_channel_not_connected_raises(self) -> None:
        from agent_mesh_router.adapters.redis_adapter import (
            RedisTransport,
            RedisTransportError,
        )
        import agent_mesh_router.adapters.redis_adapter as rm

        rm._REDIS_AVAILABLE = True
        transport = RedisTransport.__new__(RedisTransport)
        transport._url = "redis://localhost"
        transport._channel_prefix = "amr"
        transport._client = None
        transport._total_published = 0

        with pytest.raises(RedisTransportError):
            await transport.subscribe_channel("agent-x")

    @pytest.mark.asyncio
    async def test_close_calls_aclose(self) -> None:
        redis_client = self._make_redis_client()

        from agent_mesh_router.adapters.redis_adapter import RedisTransport
        import agent_mesh_router.adapters.redis_adapter as rm

        rm._REDIS_AVAILABLE = True
        transport = RedisTransport.__new__(RedisTransport)
        transport._url = "redis://localhost"
        transport._channel_prefix = "amr"
        transport._client = redis_client
        transport._total_published = 0

        await transport.close()
        redis_client.aclose.assert_called_once()
        assert transport._client is None

    def test_properties(self) -> None:
        from agent_mesh_router.adapters.redis_adapter import RedisTransport
        import agent_mesh_router.adapters.redis_adapter as rm

        rm._REDIS_AVAILABLE = True
        transport = RedisTransport.__new__(RedisTransport)
        transport._url = "redis://localhost:6379"
        transport._channel_prefix = "amr"
        transport._client = None
        transport._total_published = 42

        assert transport.url == "redis://localhost:6379"
        assert transport.total_published == 42
        assert transport.is_connected is False

    def test_repr(self) -> None:
        from agent_mesh_router.adapters.redis_adapter import RedisTransport
        import agent_mesh_router.adapters.redis_adapter as rm

        rm._REDIS_AVAILABLE = True
        transport = RedisTransport.__new__(RedisTransport)
        transport._url = "redis://myhost:6379"
        transport._channel_prefix = "amr"
        transport._client = None
        transport._total_published = 0

        assert "myhost" in repr(transport)
