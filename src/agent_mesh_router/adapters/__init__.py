"""Transport adapters for agent-mesh-router.

This package provides transport adapters that deliver ``MessageEnvelope``
objects over various network protocols:

http
    ``HttpTransport`` — HTTP POST with retry and exponential backoff.
    Requires ``httpx``.

grpc_adapter
    ``GrpcTransport`` — Abstract base class for gRPC transports.
    Requires ``grpcio``.

redis_adapter
    ``RedisTransport`` — Redis pub/sub delivery.
    Requires ``redis[asyncio]``.

kafka_adapter
    ``KafkaTransport`` — Apache Kafka producer.
    Requires ``aiokafka``.

All optional-dependency transports guard their imports with try/except so
this package is importable even when third-party dependencies are absent.
Concrete transport classes raise ``ImportError`` at construction time when
their dependency is missing.

Example
-------
::

    from agent_mesh_router.adapters import HttpTransport, RedisTransport
"""
from __future__ import annotations

from agent_mesh_router.adapters.grpc_adapter import GrpcTransport, GrpcTransportError
from agent_mesh_router.adapters.http import HttpTransport, HttpTransportError

__all__ = [
    "GrpcTransport",
    "GrpcTransportError",
    "HttpTransport",
    "HttpTransportError",
]

# Optional transports — import silently fails when deps are absent.
try:
    from agent_mesh_router.adapters.redis_adapter import (
        RedisTransport,
        RedisTransportError,
    )

    __all__ += ["RedisTransport", "RedisTransportError"]
except ImportError:
    pass

try:
    from agent_mesh_router.adapters.kafka_adapter import (
        KafkaTransport,
        KafkaTransportError,
    )

    __all__ += ["KafkaTransport", "KafkaTransportError"]
except ImportError:
    pass
