#!/usr/bin/env python3
"""Example: Resilience — Circuit Breaker and Retry Policy

Demonstrates circuit breaker state management and retry with
backoff for fault-tolerant agent communication.

Usage:
    python examples/05_resilience.py

Requirements:
    pip install agent-mesh-router
"""
from __future__ import annotations

import agent_mesh_router
from agent_mesh_router import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitState,
    RetryConfig,
    RetryExhaustedError,
    RetryPolicy,
)


def unreliable_call(attempt: int) -> str:
    """Simulates a call that succeeds only on the third attempt."""
    if attempt < 3:
        raise ConnectionError(f"timeout on attempt {attempt}")
    return "success"


def main() -> None:
    print(f"agent-mesh-router version: {agent_mesh_router.__version__}")

    # --- Circuit Breaker ---
    config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout_seconds=5,
        success_threshold=2,
    )
    breaker = CircuitBreaker(name="nlp-service", config=config)
    print(f"Circuit breaker state: {breaker.state.value}")

    # Record failures to trip the breaker
    for i in range(3):
        try:
            with breaker:
                raise ConnectionError("service unavailable")
        except ConnectionError:
            pass
    print(f"After 3 failures: {breaker.state.value}")

    # Attempt while open
    try:
        with breaker:
            pass
    except CircuitBreakerOpenError as error:
        print(f"Blocked by open circuit: {error}")

    # --- Retry Policy ---
    retry_config = RetryConfig(
        max_attempts=5,
        backoff_base_seconds=0.01,  # fast for demo
        retriable_exceptions=[ConnectionError],
    )
    policy = RetryPolicy(config=retry_config)

    attempt_counter = [0]

    def call_with_counter() -> str:
        attempt_counter[0] += 1
        return unreliable_call(attempt_counter[0])

    try:
        result = policy.execute(call_with_counter)
        print(f"\nRetry succeeded: '{result}' after {attempt_counter[0]} attempts")
    except RetryExhaustedError as error:
        print(f"Retry exhausted: {error}")

    # Demonstrate exhaustion
    attempt_counter[0] = 0
    strict_config = RetryConfig(max_attempts=2, retriable_exceptions=[ConnectionError])
    strict_policy = RetryPolicy(config=strict_config)
    try:
        strict_policy.execute(call_with_counter)
    except RetryExhaustedError as error:
        print(f"Strict retry exhausted after 2 attempts: {error}")


if __name__ == "__main__":
    main()
