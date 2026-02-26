"""TracingMiddleware — inject and propagate distributed trace context.

Every envelope that passes through the middleware receives:

1. A ``trace_id`` (if absent, a new UUID4 is generated).
2. A ``span_id`` added to ``envelope.metadata["span_id"]``.
3. A ``parent_span_id`` captured from the incoming metadata (if present)
   and recorded as ``envelope.metadata["parent_span_id"]``.
4. A ``sampled`` flag (``"1"`` or ``"0"``) controlled by ``sample_rate``.

The middleware is designed to wrap an existing async handler and can be
stacked with other middleware in a chain pattern.

Optional OpenTelemetry integration
-----------------------------------
When ``opentelemetry-sdk`` is installed the middleware can optionally emit
real OTel spans.  This is a best-effort enhancement — the middleware
functions identically without OTel.

Example
-------
::

    import asyncio
    from agent_mesh_router.messages.envelope import MessageEnvelope
    from agent_mesh_router.middleware.tracing import TracingMiddleware

    received: list[MessageEnvelope] = []

    async def my_handler(env: MessageEnvelope) -> None:
        received.append(env)

    tracing = TracingMiddleware(my_handler)

    env = MessageEnvelope(
        sender="agent-a", receiver="agent-b", payload={"cmd": "run"}
    )
    asyncio.run(tracing(env))

    assert "span_id" in received[0].metadata
    assert "sampled" in received[0].metadata
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Union

from agent_mesh_router.messages.envelope import MessageEnvelope

logger = logging.getLogger(__name__)

NextHandler = Callable[[MessageEnvelope], Union[Awaitable[None], None]]

try:
    from opentelemetry import trace as _otel_trace  # type: ignore[import-untyped]
    from opentelemetry.trace import Span  # type: ignore[import-untyped]

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False


class TracingMiddleware:
    """Inject distributed trace context into envelopes before dispatch.

    Parameters
    ----------
    next_handler:
        The downstream handler (or next middleware) to call after
        enriching the envelope.
    service_name:
        Logical name of this service for trace labelling.
    sample_rate:
        Fraction of messages to mark as sampled (0.0–1.0).
        1.0 = sample every message.  0.0 = sample none.
    use_otel:
        When True and ``opentelemetry-sdk`` is installed, create real
        OTel spans.  Silently ignored if OTel is not available.
    """

    def __init__(
        self,
        next_handler: NextHandler,
        *,
        service_name: str = "agent-mesh-router",
        sample_rate: float = 1.0,
        use_otel: bool = False,
    ) -> None:
        if not (0.0 <= sample_rate <= 1.0):
            raise ValueError(
                f"sample_rate must be in [0.0, 1.0], got {sample_rate}."
            )
        self._next = next_handler
        self._service_name = service_name
        self._sample_rate = sample_rate
        self._use_otel = use_otel and _OTEL_AVAILABLE
        self._total_processed: int = 0

        if use_otel and not _OTEL_AVAILABLE:
            logger.warning(
                "TracingMiddleware: use_otel=True but opentelemetry-sdk "
                "is not installed. OTel spans will not be emitted."
            )

    # ------------------------------------------------------------------
    # Callable interface
    # ------------------------------------------------------------------

    async def __call__(self, envelope: MessageEnvelope) -> None:
        """Enrich ``envelope`` with trace context and forward to next handler.

        Parameters
        ----------
        envelope:
            Incoming message envelope.  ``trace_id`` is preserved if already
            set; a new one is generated otherwise.
        """
        self._inject_trace_context(envelope)
        self._total_processed += 1

        if self._use_otel:
            await self._call_with_otel_span(envelope)
        else:
            await self._call_next(envelope)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def wrap(self) -> Callable[[MessageEnvelope], Awaitable[None]]:
        """Return a coroutine function usable as a broker message handler."""
        async def handler(envelope: MessageEnvelope) -> None:
            await self(envelope)

        return handler

    @property
    def total_processed(self) -> int:
        """Total envelopes processed by this middleware."""
        return self._total_processed

    @property
    def service_name(self) -> str:
        """Configured service name label."""
        return self._service_name

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _inject_trace_context(self, envelope: MessageEnvelope) -> None:
        """Mutate ``envelope.metadata`` with trace fields."""
        import random as _random

        # Ensure trace_id is set
        if not envelope.trace_id:
            envelope.trace_id = str(uuid.uuid4())

        # Generate a new span_id for this hop
        span_id = uuid.uuid4().hex[:16]  # 8-byte hex span ID

        # Carry forward parent_span_id if present
        existing_span_id = envelope.metadata.get("span_id")
        if existing_span_id:
            envelope.metadata["parent_span_id"] = existing_span_id

        envelope.metadata["span_id"] = span_id
        envelope.metadata["service_name"] = self._service_name
        envelope.metadata["sampled"] = (
            "1" if _random.random() < self._sample_rate else "0"
        )

        # Record ingestion timestamp for latency tracking
        envelope.metadata["trace_ingested_at"] = str(time.time())

    async def _call_next(self, envelope: MessageEnvelope) -> None:
        """Forward the envelope to the next handler."""
        result = self._next(envelope)
        if asyncio.isfuture(result) or asyncio.iscoroutine(result):
            await result

    async def _call_with_otel_span(self, envelope: MessageEnvelope) -> None:
        """Wrap the next handler call in an OTel span."""
        tracer = _otel_trace.get_tracer(self._service_name)
        with tracer.start_as_current_span(
            name=f"agent_mesh_router.{envelope.message_type.value}",
            attributes={
                "messaging.system": "agent-mesh-router",
                "messaging.destination": envelope.receiver,
                "messaging.message_id": envelope.message_id,
                "net.app.protocol.name": "amr",
            },
        ):
            await self._call_next(envelope)

    def __repr__(self) -> str:
        return (
            f"TracingMiddleware("
            f"service={self._service_name!r}, "
            f"sample_rate={self._sample_rate}, "
            f"otel={self._use_otel}, "
            f"processed={self._total_processed}"
            f")"
        )
