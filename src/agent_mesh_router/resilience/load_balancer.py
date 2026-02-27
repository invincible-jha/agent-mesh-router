"""LoadBalancer — distribute requests across service endpoints.

Four strategies are provided:

ROUND_ROBIN
    Cycles through healthy endpoints in stable insertion order.
WEIGHTED_ROUND_ROBIN
    Like round-robin but endpoints with higher ``weight`` receive
    proportionally more requests.
LEAST_CONNECTIONS
    Routes each call to the healthy endpoint currently handling the
    fewest active connections.
RANDOM
    Selects a healthy endpoint with uniform probability.

All methods are thread-safe.

Example
-------
::

    from agent_mesh_router.resilience.load_balancer import (
        LoadBalancer,
        LoadBalancerStrategy,
        ServiceEndpoint,
    )

    lb = LoadBalancer()
    lb.add_endpoint(ServiceEndpoint("api-1", host="10.0.0.1", port=8080))
    lb.add_endpoint(ServiceEndpoint("api-2", host="10.0.0.2", port=8080, weight=2.0))

    endpoint = lb.select(LoadBalancerStrategy.WEIGHTED_ROUND_ROBIN)
    print(endpoint.host)

    lb.mark_unhealthy("api-1")
    healthy = lb.healthy_endpoints
    print([e.name for e in healthy])  # ["api-2"]
"""
from __future__ import annotations

import logging
import random as _random
import threading
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class LoadBalancerStrategy(str, Enum):
    """Available load-balancing strategies."""

    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RANDOM = "random"


class NoHealthyEndpointError(Exception):
    """Raised when no healthy endpoints are available for selection."""


@dataclass(frozen=True)
class ServiceEndpoint:
    """Immutable descriptor for a service endpoint.

    Attributes
    ----------
    name:
        Unique logical identifier for this endpoint.
    host:
        Hostname or IP address.
    port:
        TCP port number.
    weight:
        Relative weight used by WEIGHTED_ROUND_ROBIN.  Must be > 0.
        Defaults to 1.0 (equal weight).
    healthy:
        Initial health flag.  Endpoints can be toggled at runtime via
        ``mark_healthy`` / ``mark_unhealthy``.  Defaults to True.
    """

    name: str
    host: str
    port: int
    weight: float = field(default=1.0)
    healthy: bool = field(default=True)

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError(
                f"weight must be > 0, got {self.weight} for endpoint {self.name!r}."
            )
        if not (0 < self.port <= 65535):
            raise ValueError(
                f"port must be in 1–65535, got {self.port} for endpoint {self.name!r}."
            )


class LoadBalancer:
    """Distribute requests across a pool of service endpoints.

    Parameters
    ----------
    seed:
        Optional integer seed for the internal RNG (RANDOM strategy).
        Useful for deterministic testing.
    """

    def __init__(self, *, seed: int | None = None) -> None:
        self._endpoints: dict[str, ServiceEndpoint] = {}
        # Mutable health flags kept separately so ServiceEndpoint stays frozen
        self._health: dict[str, bool] = {}
        # Mutable connection counts for LEAST_CONNECTIONS
        self._connections: dict[str, int] = {}
        # Round-robin cursor (integer counter)
        self._rr_counter: int = 0
        # Weighted round-robin cursor (float accumulator)
        self._wrr_current_weights: dict[str, float] = {}
        self._lock = threading.RLock()
        self._rng = _random.Random(seed)

    # ------------------------------------------------------------------
    # Endpoint management
    # ------------------------------------------------------------------

    def add_endpoint(self, endpoint: ServiceEndpoint) -> None:
        """Register a new endpoint.

        Re-adding an existing name replaces the endpoint configuration
        but preserves the current connection count.

        Parameters
        ----------
        endpoint:
            The endpoint descriptor to register.
        """
        with self._lock:
            existing_connections = self._connections.get(endpoint.name, 0)
            self._endpoints[endpoint.name] = endpoint
            self._health[endpoint.name] = endpoint.healthy
            self._connections[endpoint.name] = existing_connections
            self._wrr_current_weights[endpoint.name] = 0.0
        logger.debug("LoadBalancer: added endpoint %r.", endpoint.name)

    def remove_endpoint(self, name: str) -> None:
        """Remove an endpoint from the pool.

        Silently ignores names that are not registered.

        Parameters
        ----------
        name:
            The endpoint name to remove.
        """
        with self._lock:
            self._endpoints.pop(name, None)
            self._health.pop(name, None)
            self._connections.pop(name, None)
            self._wrr_current_weights.pop(name, None)
        logger.debug("LoadBalancer: removed endpoint %r.", name)

    def mark_unhealthy(self, name: str) -> None:
        """Mark an endpoint as unhealthy, excluding it from selection.

        Parameters
        ----------
        name:
            The endpoint name to mark.

        Raises
        ------
        KeyError
            If the endpoint is not registered.
        """
        with self._lock:
            if name not in self._endpoints:
                raise KeyError(f"Endpoint {name!r} is not registered.")
            self._health[name] = False
        logger.info("LoadBalancer: endpoint %r marked UNHEALTHY.", name)

    def mark_healthy(self, name: str) -> None:
        """Mark an endpoint as healthy, re-including it in selection.

        Parameters
        ----------
        name:
            The endpoint name to mark.

        Raises
        ------
        KeyError
            If the endpoint is not registered.
        """
        with self._lock:
            if name not in self._endpoints:
                raise KeyError(f"Endpoint {name!r} is not registered.")
            self._health[name] = True
        logger.info("LoadBalancer: endpoint %r marked HEALTHY.", name)

    # ------------------------------------------------------------------
    # Connection tracking
    # ------------------------------------------------------------------

    def record_connection_open(self, name: str) -> None:
        """Increment the active connection count for an endpoint.

        Used with LEAST_CONNECTIONS strategy.

        Parameters
        ----------
        name:
            Endpoint name.

        Raises
        ------
        KeyError
            If the endpoint is not registered.
        """
        with self._lock:
            if name not in self._connections:
                raise KeyError(f"Endpoint {name!r} is not registered.")
            self._connections[name] += 1

    def record_connection_close(self, name: str) -> None:
        """Decrement the active connection count for an endpoint.

        The count will not go below zero.

        Parameters
        ----------
        name:
            Endpoint name.

        Raises
        ------
        KeyError
            If the endpoint is not registered.
        """
        with self._lock:
            if name not in self._connections:
                raise KeyError(f"Endpoint {name!r} is not registered.")
            self._connections[name] = max(0, self._connections[name] - 1)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(
        self,
        strategy: LoadBalancerStrategy = LoadBalancerStrategy.ROUND_ROBIN,
    ) -> ServiceEndpoint:
        """Select an endpoint according to the given strategy.

        Parameters
        ----------
        strategy:
            The algorithm to use.  Defaults to ``ROUND_ROBIN``.

        Returns
        -------
        ServiceEndpoint
            A healthy endpoint chosen by the strategy.

        Raises
        ------
        NoHealthyEndpointError
            If no healthy endpoints are registered.
        """
        with self._lock:
            healthy = self._healthy_list()
            if not healthy:
                raise NoHealthyEndpointError(
                    "No healthy endpoints are available."
                )

            match strategy:
                case LoadBalancerStrategy.ROUND_ROBIN:
                    return self._round_robin(healthy)
                case LoadBalancerStrategy.WEIGHTED_ROUND_ROBIN:
                    return self._weighted_round_robin(healthy)
                case LoadBalancerStrategy.LEAST_CONNECTIONS:
                    return self._least_connections(healthy)
                case LoadBalancerStrategy.RANDOM:
                    return self._random(healthy)
                case _:  # pragma: no cover
                    raise ValueError(f"Unknown strategy: {strategy!r}")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def healthy_endpoints(self) -> list[ServiceEndpoint]:
        """Return a snapshot of all currently healthy endpoints."""
        with self._lock:
            return self._healthy_list()

    @property
    def all_endpoints(self) -> list[ServiceEndpoint]:
        """Return a snapshot of all registered endpoints (healthy and not)."""
        with self._lock:
            return list(self._endpoints.values())

    # ------------------------------------------------------------------
    # Strategy implementations (all called with lock held)
    # ------------------------------------------------------------------

    def _healthy_list(self) -> list[ServiceEndpoint]:
        """Return healthy endpoints in stable insertion order."""
        return [
            ep
            for name, ep in self._endpoints.items()
            if self._health.get(name, False)
        ]

    def _round_robin(self, healthy: list[ServiceEndpoint]) -> ServiceEndpoint:
        index = self._rr_counter % len(healthy)
        self._rr_counter += 1
        return healthy[index]

    def _weighted_round_robin(
        self, healthy: list[ServiceEndpoint]
    ) -> ServiceEndpoint:
        """Smooth weighted round-robin (Nginx algorithm).

        Each endpoint's ``current_weight`` increases by its ``weight`` each
        round, then the winner's weight decreases by the total weight.
        This produces a smooth distribution without integer expansion.
        """
        total_weight = sum(ep.weight for ep in healthy)
        best: ServiceEndpoint | None = None
        best_weight = float("-inf")

        for ep in healthy:
            self._wrr_current_weights[ep.name] = (
                self._wrr_current_weights.get(ep.name, 0.0) + ep.weight
            )
            if self._wrr_current_weights[ep.name] > best_weight:
                best_weight = self._wrr_current_weights[ep.name]
                best = ep

        assert best is not None  # healthy is non-empty
        self._wrr_current_weights[best.name] -= total_weight
        return best

    def _least_connections(
        self, healthy: list[ServiceEndpoint]
    ) -> ServiceEndpoint:
        return min(healthy, key=lambda ep: self._connections.get(ep.name, 0))

    def _random(self, healthy: list[ServiceEndpoint]) -> ServiceEndpoint:
        return self._rng.choice(healthy)

    def __repr__(self) -> str:
        with self._lock:
            total = len(self._endpoints)
            healthy_count = sum(1 for v in self._health.values() if v)
        return (
            f"LoadBalancer("
            f"endpoints={total}, "
            f"healthy={healthy_count}"
            f")"
        )
