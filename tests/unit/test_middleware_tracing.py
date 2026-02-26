"""Tests for TracingMiddleware."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_mesh_router.messages.envelope import MessageEnvelope
from agent_mesh_router.middleware.tracing import TracingMiddleware


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _envelope() -> MessageEnvelope:
    return MessageEnvelope(
        sender="sender-a",
        receiver="receiver-b",
        payload={"cmd": "run"},
    )


async def _noop(env: MessageEnvelope) -> None:
    """No-op async handler for testing."""


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestTracingMiddlewareConstruction:
    def test_valid_construction_defaults(self) -> None:
        tm = TracingMiddleware(_noop)
        assert tm.service_name == "agent-mesh-router"
        assert tm.total_processed == 0

    def test_custom_service_name(self) -> None:
        tm = TracingMiddleware(_noop, service_name="my-service")
        assert tm.service_name == "my-service"

    def test_invalid_sample_rate_too_high(self) -> None:
        with pytest.raises(ValueError):
            TracingMiddleware(_noop, sample_rate=1.1)

    def test_invalid_sample_rate_negative(self) -> None:
        with pytest.raises(ValueError):
            TracingMiddleware(_noop, sample_rate=-0.1)

    def test_sample_rate_zero_is_valid(self) -> None:
        tm = TracingMiddleware(_noop, sample_rate=0.0)
        assert tm.total_processed == 0

    def test_sample_rate_one_is_valid(self) -> None:
        tm = TracingMiddleware(_noop, sample_rate=1.0)
        assert tm.total_processed == 0

    def test_repr_contains_service_name(self) -> None:
        tm = TracingMiddleware(_noop, service_name="svc-x")
        assert "svc-x" in repr(tm)

    def test_repr_contains_processed_count(self) -> None:
        tm = TracingMiddleware(_noop)
        assert "processed=0" in repr(tm)


# ---------------------------------------------------------------------------
# Trace context injection
# ---------------------------------------------------------------------------


class TestTraceContextInjection:
    @pytest.mark.asyncio
    async def test_span_id_injected(self) -> None:
        received: list[MessageEnvelope] = []

        async def capture(env: MessageEnvelope) -> None:
            received.append(env)

        tm = TracingMiddleware(capture)
        env = _envelope()
        await tm(env)
        assert "span_id" in received[0].metadata

    @pytest.mark.asyncio
    async def test_sampled_field_injected(self) -> None:
        received: list[MessageEnvelope] = []

        async def capture(env: MessageEnvelope) -> None:
            received.append(env)

        tm = TracingMiddleware(capture, sample_rate=1.0)
        await tm(_envelope())
        assert received[0].metadata.get("sampled") == "1"

    @pytest.mark.asyncio
    async def test_sample_rate_zero_marks_unsampled(self) -> None:
        received: list[MessageEnvelope] = []

        async def capture(env: MessageEnvelope) -> None:
            received.append(env)

        tm = TracingMiddleware(capture, sample_rate=0.0)
        await tm(_envelope())
        assert received[0].metadata.get("sampled") == "0"

    @pytest.mark.asyncio
    async def test_trace_id_preserved_when_already_set(self) -> None:
        """Envelopes already have a trace_id; the middleware preserves it."""
        received: list[MessageEnvelope] = []

        async def capture(env: MessageEnvelope) -> None:
            received.append(env)

        tm = TracingMiddleware(capture)
        env = _envelope()
        original_trace_id = env.trace_id
        await tm(env)
        # trace_id must still be present (same value since it was already set).
        assert received[0].trace_id == original_trace_id

    @pytest.mark.asyncio
    async def test_existing_trace_id_preserved(self) -> None:
        received: list[MessageEnvelope] = []

        async def capture(env: MessageEnvelope) -> None:
            received.append(env)

        tm = TracingMiddleware(capture)
        env = _envelope()
        env.trace_id = "my-custom-trace-id"
        await tm(env)
        assert received[0].trace_id == "my-custom-trace-id"

    @pytest.mark.asyncio
    async def test_parent_span_id_propagated(self) -> None:
        received: list[MessageEnvelope] = []

        async def capture(env: MessageEnvelope) -> None:
            received.append(env)

        tm = TracingMiddleware(capture)
        env = _envelope()
        env.metadata["span_id"] = "previous-span"
        await tm(env)
        assert received[0].metadata.get("parent_span_id") == "previous-span"
        # New span_id should differ from the previous one.
        assert received[0].metadata.get("span_id") != "previous-span"

    @pytest.mark.asyncio
    async def test_service_name_in_metadata(self) -> None:
        received: list[MessageEnvelope] = []

        async def capture(env: MessageEnvelope) -> None:
            received.append(env)

        tm = TracingMiddleware(capture, service_name="svc-z")
        await tm(_envelope())
        assert received[0].metadata.get("service_name") == "svc-z"

    @pytest.mark.asyncio
    async def test_trace_ingested_at_set(self) -> None:
        received: list[MessageEnvelope] = []

        async def capture(env: MessageEnvelope) -> None:
            received.append(env)

        tm = TracingMiddleware(capture)
        await tm(_envelope())
        assert "trace_ingested_at" in received[0].metadata

    @pytest.mark.asyncio
    async def test_total_processed_increments(self) -> None:
        tm = TracingMiddleware(_noop)
        await tm(_envelope())
        await tm(_envelope())
        assert tm.total_processed == 2


# ---------------------------------------------------------------------------
# Sync and async next_handler support
# ---------------------------------------------------------------------------


class TestNextHandlerVariants:
    @pytest.mark.asyncio
    async def test_sync_handler_called(self) -> None:
        called: list[bool] = []

        def sync_handler(env: MessageEnvelope) -> None:
            called.append(True)

        tm = TracingMiddleware(sync_handler)
        await tm(_envelope())
        assert called == [True]

    @pytest.mark.asyncio
    async def test_async_handler_called(self) -> None:
        called: list[bool] = []

        async def async_handler(env: MessageEnvelope) -> None:
            called.append(True)

        tm = TracingMiddleware(async_handler)
        await tm(_envelope())
        assert called == [True]


# ---------------------------------------------------------------------------
# wrap() helper
# ---------------------------------------------------------------------------


class TestWrapHelper:
    @pytest.mark.asyncio
    async def test_wrap_returns_callable(self) -> None:
        tm = TracingMiddleware(_noop)
        wrapped = tm.wrap()
        env = _envelope()
        await wrapped(env)
        assert tm.total_processed == 1
