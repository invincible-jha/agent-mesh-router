"""Middleware pipeline for agent-mesh-router envelope processing.

Middleware components sit between the broker and agent handlers, enriching
or transforming envelopes as they pass through.

Available middleware
--------------------
tracing
    ``TracingMiddleware`` — injects ``trace_id``, ``span_id``, and sampling
    flags into ``envelope.metadata`` for distributed tracing.  Optionally
    emits OpenTelemetry spans when ``opentelemetry-sdk`` is installed.

Example: wrapping a handler
---------------------------
::

    from agent_mesh_router.middleware import TracingMiddleware

    async def my_handler(envelope):
        print(f"Got: {envelope.message_id}")

    tracing = TracingMiddleware(my_handler, sample_rate=0.5)

    # Use tracing.wrap() as the broker handler:
    await broker.subscribe("my-agent", tracing.wrap())
"""
from __future__ import annotations

from agent_mesh_router.middleware.tracing import TracingMiddleware

__all__ = ["TracingMiddleware"]
