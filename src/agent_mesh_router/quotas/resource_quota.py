"""Per-agent resource quota definitions and enforcement.

Design
------
Each agent is assigned a :class:`ResourceQuota` that caps one or more
resource dimensions (tokens, memory, CPU, API calls).  The
:class:`QuotaEnforcer` maintains a usage counter per agent per dimension
and rejects requests that would push usage beyond the configured limits.

All limits use ``None`` to mean "unlimited" for that dimension.

Thread Safety
-------------
:class:`QuotaEnforcer` is protected by a single :class:`threading.Lock`
so that the check-and-record operation is atomic.

Usage
-----
::

    from agent_mesh_router.quotas import ResourceQuota, QuotaEnforcer

    quota = ResourceQuota(agent_id="agent_a", token_limit=10_000, api_call_limit=100)
    enforcer = QuotaEnforcer()
    enforcer.register(quota)

    # Check before acting
    ok = enforcer.check("agent_a", tokens=500, api_calls=1)
    if ok:
        enforcer.record("agent_a", tokens=500, api_calls=1)
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class QuotaViolation(RuntimeError):
    """Raised when a resource request would exceed the agent's quota.

    Attributes
    ----------
    agent_id:
        The agent whose quota would be violated.
    dimension:
        The resource dimension that triggered the violation.
    requested:
        Amount requested.
    remaining:
        Amount remaining before the limit.
    limit:
        The configured limit for the dimension.
    """

    def __init__(
        self,
        agent_id: str,
        dimension: str,
        requested: float,
        remaining: float,
        limit: float,
    ) -> None:
        self.agent_id = agent_id
        self.dimension = dimension
        self.requested = requested
        self.remaining = remaining
        self.limit = limit
        super().__init__(
            f"Quota violation for agent '{agent_id}': "
            f"dimension={dimension!r}, requested={requested}, "
            f"remaining={remaining}, limit={limit}."
        )


# ---------------------------------------------------------------------------
# ResourceQuota
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResourceQuota:
    """Per-agent resource limits across multiple dimensions.

    All limit fields default to ``None`` (unlimited).

    Parameters
    ----------
    agent_id:
        Unique identifier of the agent these limits apply to.
    token_limit:
        Maximum tokens (input + output combined) the agent may consume.
    memory_limit_mb:
        Maximum memory in megabytes.
    cpu_limit_seconds:
        Maximum CPU time in seconds.
    api_call_limit:
        Maximum number of API/tool calls.
    label:
        Human-readable label for logging and reporting.
    """

    agent_id: str
    token_limit: float | None = None
    memory_limit_mb: float | None = None
    cpu_limit_seconds: float | None = None
    api_call_limit: float | None = None
    label: str = ""

    def __post_init__(self) -> None:
        if not self.agent_id.strip():
            raise ValueError("agent_id must not be empty.")
        for name, value in [
            ("token_limit", self.token_limit),
            ("memory_limit_mb", self.memory_limit_mb),
            ("cpu_limit_seconds", self.cpu_limit_seconds),
            ("api_call_limit", self.api_call_limit),
        ]:
            if value is not None and value < 0:
                raise ValueError(
                    f"{name} must be non-negative or None, got {value!r}."
                )

    @property
    def dimensions(self) -> dict[str, float | None]:
        """Return all configured limits as a dimension -> limit mapping."""
        return {
            "tokens": self.token_limit,
            "memory_mb": self.memory_limit_mb,
            "cpu_seconds": self.cpu_limit_seconds,
            "api_calls": self.api_call_limit,
        }


# ---------------------------------------------------------------------------
# QuotaUsage (snapshot)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuotaUsage:
    """Point-in-time snapshot of an agent's resource consumption.

    Parameters
    ----------
    agent_id:
        The agent being reported on.
    tokens_used:
        Total tokens consumed in this period.
    memory_mb_used:
        Total memory used in megabytes.
    cpu_seconds_used:
        Total CPU seconds consumed.
    api_calls_used:
        Total API calls made.
    snapshotted_at:
        When this snapshot was taken (UTC).
    """

    agent_id: str
    tokens_used: float
    memory_mb_used: float
    cpu_seconds_used: float
    api_calls_used: float
    snapshotted_at: datetime

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dictionary."""
        return {
            "agent_id": self.agent_id,
            "tokens_used": self.tokens_used,
            "memory_mb_used": self.memory_mb_used,
            "cpu_seconds_used": self.cpu_seconds_used,
            "api_calls_used": self.api_calls_used,
            "snapshotted_at": self.snapshotted_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# QuotaEnforcer
# ---------------------------------------------------------------------------


class QuotaEnforcer:
    """Register quotas and enforce resource usage for a set of agents.

    Usage is tracked across four dimensions: tokens, memory_mb,
    cpu_seconds, and api_calls.

    All public methods are thread-safe.

    Example
    -------
    ::

        enforcer = QuotaEnforcer()
        enforcer.register(ResourceQuota("agent_a", token_limit=1000))
        enforcer.record("agent_a", tokens=200)
        usage = enforcer.usage("agent_a")
        assert usage.tokens_used == 200
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # agent_id -> ResourceQuota
        self._quotas: dict[str, ResourceQuota] = {}
        # agent_id -> {dimension: float}
        self._usage: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, quota: ResourceQuota) -> None:
        """Register (or replace) the quota for an agent.

        Parameters
        ----------
        quota:
            The :class:`ResourceQuota` to register.
        """
        with self._lock:
            self._quotas[quota.agent_id] = quota
            if quota.agent_id not in self._usage:
                self._usage[quota.agent_id] = _zero_usage()

    def unregister(self, agent_id: str) -> bool:
        """Remove the quota and usage record for *agent_id*.

        Returns
        -------
        bool
            True if the agent was registered; False otherwise.
        """
        with self._lock:
            if agent_id not in self._quotas:
                return False
            del self._quotas[agent_id]
            self._usage.pop(agent_id, None)
            return True

    # ------------------------------------------------------------------
    # Check and record
    # ------------------------------------------------------------------

    def check(
        self,
        agent_id: str,
        tokens: float = 0.0,
        memory_mb: float = 0.0,
        cpu_seconds: float = 0.0,
        api_calls: float = 0.0,
        raise_on_violation: bool = False,
    ) -> bool:
        """Check whether the given resource usage would violate the quota.

        Parameters
        ----------
        agent_id:
            Target agent.
        tokens:
            Additional tokens to check.
        memory_mb:
            Additional memory (MB) to check.
        cpu_seconds:
            Additional CPU seconds to check.
        api_calls:
            Additional API calls to check.
        raise_on_violation:
            If True, raise :class:`QuotaViolation` instead of returning
            False when a dimension would be exceeded.

        Returns
        -------
        bool
            True when all dimensions are within quota.

        Raises
        ------
        KeyError
            If *agent_id* is not registered.
        QuotaViolation
            If *raise_on_violation* is True and a limit would be exceeded.
        """
        with self._lock:
            if agent_id not in self._quotas:
                raise KeyError(f"Agent '{agent_id}' is not registered.")
            quota = self._quotas[agent_id]
            current = self._usage[agent_id]
            requests = {
                "tokens": tokens,
                "memory_mb": memory_mb,
                "cpu_seconds": cpu_seconds,
                "api_calls": api_calls,
            }
            limits = {
                "tokens": quota.token_limit,
                "memory_mb": quota.memory_limit_mb,
                "cpu_seconds": quota.cpu_limit_seconds,
                "api_calls": quota.api_call_limit,
            }
            for dimension, requested in requests.items():
                if requested <= 0:
                    continue
                limit = limits[dimension]
                if limit is None:
                    continue
                used = current[dimension]
                remaining = max(0.0, limit - used)
                if requested > remaining:
                    if raise_on_violation:
                        raise QuotaViolation(
                            agent_id=agent_id,
                            dimension=dimension,
                            requested=requested,
                            remaining=remaining,
                            limit=limit,
                        )
                    return False
        return True

    def record(
        self,
        agent_id: str,
        tokens: float = 0.0,
        memory_mb: float = 0.0,
        cpu_seconds: float = 0.0,
        api_calls: float = 0.0,
    ) -> None:
        """Record actual resource consumption for *agent_id*.

        Parameters
        ----------
        agent_id:
            Target agent.
        tokens:
            Tokens consumed.
        memory_mb:
            Memory consumed in MB.
        cpu_seconds:
            CPU seconds consumed.
        api_calls:
            API calls made.

        Raises
        ------
        KeyError
            If *agent_id* is not registered.
        ValueError
            If any value is negative.
        """
        for name, value in [
            ("tokens", tokens),
            ("memory_mb", memory_mb),
            ("cpu_seconds", cpu_seconds),
            ("api_calls", api_calls),
        ]:
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value!r}.")

        with self._lock:
            if agent_id not in self._quotas:
                raise KeyError(f"Agent '{agent_id}' is not registered.")
            current = self._usage[agent_id]
            current["tokens"] += tokens
            current["memory_mb"] += memory_mb
            current["cpu_seconds"] += cpu_seconds
            current["api_calls"] += api_calls

    def check_and_record(
        self,
        agent_id: str,
        tokens: float = 0.0,
        memory_mb: float = 0.0,
        cpu_seconds: float = 0.0,
        api_calls: float = 0.0,
        raise_on_violation: bool = True,
    ) -> bool:
        """Atomically check quota and record usage if allowed.

        Parameters
        ----------
        agent_id:
            Target agent.
        tokens:
            Tokens to check and record.
        memory_mb:
            Memory (MB) to check and record.
        cpu_seconds:
            CPU seconds to check and record.
        api_calls:
            API calls to check and record.
        raise_on_violation:
            If True (default), raise :class:`QuotaViolation` on breach.

        Returns
        -------
        bool
            True when all dimensions are within quota and usage was recorded.

        Raises
        ------
        QuotaViolation
            If *raise_on_violation* is True and a limit would be exceeded.
        """
        with self._lock:
            if agent_id not in self._quotas:
                raise KeyError(f"Agent '{agent_id}' is not registered.")
            quota = self._quotas[agent_id]
            current = self._usage[agent_id]
            requests = {
                "tokens": tokens,
                "memory_mb": memory_mb,
                "cpu_seconds": cpu_seconds,
                "api_calls": api_calls,
            }
            limits = {
                "tokens": quota.token_limit,
                "memory_mb": quota.memory_limit_mb,
                "cpu_seconds": quota.cpu_limit_seconds,
                "api_calls": quota.api_call_limit,
            }
            for dimension, requested in requests.items():
                if requested <= 0:
                    continue
                limit = limits[dimension]
                if limit is None:
                    continue
                used = current[dimension]
                remaining = max(0.0, limit - used)
                if requested > remaining:
                    if raise_on_violation:
                        raise QuotaViolation(
                            agent_id=agent_id,
                            dimension=dimension,
                            requested=requested,
                            remaining=remaining,
                            limit=limit,
                        )
                    return False

            # All checks passed — record
            current["tokens"] += tokens
            current["memory_mb"] += memory_mb
            current["cpu_seconds"] += cpu_seconds
            current["api_calls"] += api_calls
        return True

    def reset(self, agent_id: str) -> None:
        """Reset usage counters for *agent_id* to zero.

        Raises
        ------
        KeyError
            If *agent_id* is not registered.
        """
        with self._lock:
            if agent_id not in self._quotas:
                raise KeyError(f"Agent '{agent_id}' is not registered.")
            self._usage[agent_id] = _zero_usage()

    def reset_all(self) -> None:
        """Reset usage counters for all registered agents."""
        with self._lock:
            for agent_id in self._usage:
                self._usage[agent_id] = _zero_usage()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def usage(self, agent_id: str) -> QuotaUsage:
        """Return a snapshot of usage for *agent_id*.

        Raises
        ------
        KeyError
            If *agent_id* is not registered.
        """
        with self._lock:
            if agent_id not in self._quotas:
                raise KeyError(f"Agent '{agent_id}' is not registered.")
            current = self._usage[agent_id]
            return QuotaUsage(
                agent_id=agent_id,
                tokens_used=current["tokens"],
                memory_mb_used=current["memory_mb"],
                cpu_seconds_used=current["cpu_seconds"],
                api_calls_used=current["api_calls"],
                snapshotted_at=datetime.now(timezone.utc),
            )

    def quota(self, agent_id: str) -> ResourceQuota:
        """Return the registered quota for *agent_id*.

        Raises
        ------
        KeyError
            If *agent_id* is not registered.
        """
        with self._lock:
            if agent_id not in self._quotas:
                raise KeyError(f"Agent '{agent_id}' is not registered.")
            return self._quotas[agent_id]

    def registered_agents(self) -> list[str]:
        """Return sorted list of registered agent IDs."""
        with self._lock:
            return sorted(self._quotas.keys())

    def __len__(self) -> int:
        """Return the number of registered agents."""
        with self._lock:
            return len(self._quotas)

    def __contains__(self, agent_id: object) -> bool:
        """Return True if *agent_id* is registered."""
        with self._lock:
            return agent_id in self._quotas


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _zero_usage() -> dict[str, float]:
    return {
        "tokens": 0.0,
        "memory_mb": 0.0,
        "cpu_seconds": 0.0,
        "api_calls": 0.0,
    }
