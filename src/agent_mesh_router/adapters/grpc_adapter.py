"""GrpcTransport — abstract stub for gRPC envelope delivery.

This module defines the abstract interface that a concrete gRPC transport
must implement.  A full implementation requires a generated protobuf stub
and an active gRPC channel, which are beyond the scope of this library's
core dependencies.

Third-party integrators should subclass ``GrpcTransport``, inject their
generated stub (e.g. ``AgentMeshStub``), and implement ``send``.

Optional dependency
-------------------
Requires ``grpcio`` and ``grpcio-tools``.  Import is guarded.

Install with::

    pip install agent-mesh-router[grpc]

Example
-------
::

    # In your integration package:
    import grpc
    from agent_mesh_router.adapters.grpc_adapter import GrpcTransport
    from my_proto import AgentMeshStub, EnvelopeProto

    class MyGrpcTransport(GrpcTransport):
        def __init__(self, target: str) -> None:
            channel = grpc.aio.insecure_channel(target)
            self._stub = AgentMeshStub(channel)

        async def send(self, envelope):
            proto = EnvelopeProto(...)
            return await self._stub.Deliver(proto)
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from agent_mesh_router.messages.envelope import MessageEnvelope

logger = logging.getLogger(__name__)

try:
    import grpc  # noqa: F401

    _GRPC_AVAILABLE = True
except ImportError:
    _GRPC_AVAILABLE = False


class GrpcTransportError(RuntimeError):
    """Raised when a gRPC envelope delivery fails permanently."""


class GrpcTransport(ABC):
    """Abstract base class for gRPC-based envelope transports.

    Subclasses must implement ``send`` using a concrete gRPC stub.
    The ``connect`` / ``close`` lifecycle methods are optional hooks
    for managing channel setup and teardown.
    """

    @abstractmethod
    async def send(self, envelope: MessageEnvelope) -> dict[str, object]:
        """Deliver an envelope over a gRPC channel.

        Parameters
        ----------
        envelope:
            The envelope to transmit.

        Returns
        -------
        dict[str, object]
            Response payload from the remote agent (deserialized from proto).

        Raises
        ------
        GrpcTransportError
            If the RPC fails after any retries.
        """

    async def connect(self) -> None:
        """Establish the gRPC channel connection.

        Override in subclasses that require explicit channel lifecycle
        management (e.g. ``grpc.aio`` channels).  The default is a no-op.
        """

    async def close(self) -> None:
        """Gracefully close the gRPC channel.

        Override in subclasses.  The default is a no-op.
        """

    @staticmethod
    def assert_grpc_available() -> None:
        """Raise ImportError if grpcio is not installed.

        Call this at the top of concrete implementations to produce a
        clear, actionable error at construction time rather than at
        the first RPC call.

        Raises
        ------
        ImportError
            If ``grpcio`` cannot be imported.
        """
        if not _GRPC_AVAILABLE:
            raise ImportError(
                "GrpcTransport requires 'grpcio'. "
                "Install it with: pip install agent-mesh-router[grpc]"
            )
