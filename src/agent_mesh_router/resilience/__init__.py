"""Resilience layer — circuit breakers, retries, load balancing, and health checks.

This package provides building blocks for fault-tolerant service mesh
communication:

circuit_breaker
    ``CircuitBreaker`` — wraps callables and opens the circuit when failures
    exceed a threshold, preventing cascading failures to downstream services.

retry
    ``RetryPolicy`` — retries failed callables with configurable exponential
    backoff and optional jitter.

load_balancer
    ``LoadBalancer`` — distributes requests across a pool of ``ServiceEndpoint``
    instances using round-robin, weighted round-robin, least-connections,
    or random strategies.

health_check
    ``HealthChecker`` — runs probe callables against endpoints and maintains
    per-endpoint result history with threshold-based status transitions.

Example
-------
::

    from agent_mesh_router.resilience import (
        CircuitBreaker,
        CircuitBreakerConfig,
        CircuitBreakerOpenError,
        CircuitState,
        RetryPolicy,
        RetryConfig,
        RetryExhaustedError,
        LoadBalancer,
        LoadBalancerStrategy,
        ServiceEndpoint,
        NoHealthyEndpointError,
        HealthChecker,
        HealthCheckConfig,
        HealthCheckResult,
        HealthStatus,
    )
"""
from __future__ import annotations

from agent_mesh_router.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitState,
)
from agent_mesh_router.resilience.health_check import (
    HealthCheckConfig,
    HealthCheckResult,
    HealthChecker,
    HealthStatus,
)
from agent_mesh_router.resilience.load_balancer import (
    LoadBalancer,
    LoadBalancerStrategy,
    NoHealthyEndpointError,
    ServiceEndpoint,
)
from agent_mesh_router.resilience.retry import (
    RetryConfig,
    RetryExhaustedError,
    RetryPolicy,
)

__all__: list[str] = [
    # circuit_breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerOpenError",
    "CircuitState",
    # retry
    "RetryConfig",
    "RetryExhaustedError",
    "RetryPolicy",
    # load_balancer
    "LoadBalancer",
    "LoadBalancerStrategy",
    "NoHealthyEndpointError",
    "ServiceEndpoint",
    # health_check
    "HealthCheckConfig",
    "HealthCheckResult",
    "HealthChecker",
    "HealthStatus",
]
