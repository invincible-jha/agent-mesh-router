"""AumOS Kubernetes Operator — entry point.

Registers all kopf handlers from the agent and mesh controllers, configures
structured logging, and starts the OpenTelemetry SDK before the event loop
takes over.

Usage (local development)::

    pip install -r requirements.txt
    python -m kopf run main.py --namespace aumos-system --dev

Usage (in-cluster, via Dockerfile)::

    docker run --rm \\
        -e KUBERNETES_SERVICE_HOST=... \\
        -e KUBERNETES_SERVICE_PORT=... \\
        ghcr.io/aumos-ai/aumos-operator:latest
"""
from __future__ import annotations

import os
import logging

import kopf
import structlog

# Import controllers so that their @kopf.on.* decorators are registered.
from controllers import agent_controller  # noqa: F401
from controllers import mesh_controller  # noqa: F401

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _configure_logging() -> None:
    """Wire structlog to Python's standard logging at INFO level."""
    log_level_name: str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level: int = getattr(logging, log_level_name, logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(level=log_level)


# ---------------------------------------------------------------------------
# OpenTelemetry (optional — skipped when OTEL_EXPORTER_OTLP_ENDPOINT is unset)
# ---------------------------------------------------------------------------

def _configure_otel() -> None:
    """Initialise OTLP trace exporter when the endpoint env-var is present."""
    endpoint: str | None = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource

        service_name: str = os.getenv("OTEL_SERVICE_NAME", "aumos-operator")
        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        structlog.get_logger().info(
            "otel_configured",
            endpoint=endpoint,
            service_name=service_name,
        )
    except ImportError:
        structlog.get_logger().warning(
            "otel_unavailable",
            reason="opentelemetry packages not installed",
        )


# ---------------------------------------------------------------------------
# kopf lifecycle hooks
# ---------------------------------------------------------------------------

@kopf.on.startup()
def on_startup(settings: kopf.OperatorSettings, **_kwargs: object) -> None:
    """Operator startup hook — configure settings and log boot message."""
    _configure_logging()
    _configure_otel()

    # Tune reconciliation behaviour.
    settings.persistence.finalizer = "aumos.ai/operator-finalizer"
    settings.posting.level = logging.WARNING  # only post WARNING+ events to K8s
    settings.watching.server_timeout = int(os.getenv("WATCH_TIMEOUT", "300"))

    structlog.get_logger().info(
        "aumos_operator_started",
        version="0.1.0",
        namespace=os.getenv("WATCH_NAMESPACE", "all"),
    )


@kopf.on.cleanup()
def on_cleanup(**_kwargs: object) -> None:
    """Operator shutdown hook — flush telemetry before exit."""
    try:
        from opentelemetry import trace
        provider = trace.get_tracer_provider()
        if hasattr(provider, "shutdown"):
            provider.shutdown()  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass

    structlog.get_logger().info("aumos_operator_stopped")
