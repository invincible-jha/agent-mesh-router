"""Fallback handler registry for circuit breaker integration.

Provides a pluggable registry of fallback handlers that are invoked
when a circuit breaker is open and a downstream call cannot proceed.
Handlers are keyed by service name and can return static values,
cached last-known-good responses, or delegate to other services.

Classes
-------
- FallbackHandler        Abstract base for all fallback handlers.
- StaticFallbackHandler  Returns a preconfigured static value.
- CacheFallbackHandler   Returns the last successfully cached response.
- FallbackRegistry       Registry keyed by service name.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class FallbackHandler(ABC):
    """Base class for circuit breaker fallback handlers.

    All concrete handlers must implement :meth:`handle`.
    """

    @abstractmethod
    def handle(
        self,
        service_name: str,
        error: Exception,
        context: dict[str, object],
    ) -> object:
        """Execute the fallback and return a substitute response.

        Parameters
        ----------
        service_name:
            Name of the service whose circuit opened.
        error:
            The exception that caused the circuit to open or the
            ``CircuitBreakerOpenError`` itself.
        context:
            Caller-supplied key/value pairs that a handler may use to
            tailor the fallback (e.g. request payload, user id).

        Returns
        -------
        object
            A substitute response to return to the caller.  The exact
            type is handler-defined and must be understood by the caller.
        """


# ---------------------------------------------------------------------------
# Concrete handlers
# ---------------------------------------------------------------------------


class StaticFallbackHandler(FallbackHandler):
    """Returns a preconfigured static value for every fallback invocation.

    Useful when a safe default (empty list, zero, placeholder string, etc.)
    can always stand in for a failed service call.

    Parameters
    ----------
    default_value:
        The value returned unconditionally by :meth:`handle`.
    """

    def __init__(self, default_value: object) -> None:
        self._default_value = default_value

    @property
    def default_value(self) -> object:
        """The static value returned by this handler."""
        return self._default_value

    def handle(
        self,
        service_name: str,
        error: Exception,
        context: dict[str, object],
    ) -> object:
        """Return the preconfigured static value.

        Parameters
        ----------
        service_name:
            Ignored — the same value is returned regardless.
        error:
            Logged at DEBUG level; otherwise ignored.
        context:
            Ignored.

        Returns
        -------
        object
            The static default value supplied at construction time.
        """
        logger.debug(
            "StaticFallbackHandler: service=%r error=%r — returning static default.",
            service_name,
            str(error),
        )
        return self._default_value


class CacheFallbackHandler(FallbackHandler):
    """Returns the last successfully cached response for a service.

    The cache is populated via :meth:`update_cache`.  If no cached value
    exists for the requested service a ``KeyError`` is raised so the
    caller can decide what to do (register a chain fallback, re-raise,
    etc.).

    The internal cache is a plain dict — no TTL or eviction logic is
    applied intentionally.  Advanced caching belongs in a separate layer.
    """

    def __init__(self) -> None:
        self._cache: dict[str, object] = {}

    def update_cache(self, service_name: str, value: object) -> None:
        """Store *value* as the latest successful response for *service_name*.

        Parameters
        ----------
        service_name:
            Service identifier matching the key used in the registry.
        value:
            Successful response to cache.
        """
        self._cache[service_name] = value
        logger.debug("CacheFallbackHandler: cached response for service=%r.", service_name)

    def has_cached_value(self, service_name: str) -> bool:
        """Return True if a cached value exists for *service_name*."""
        return service_name in self._cache

    def handle(
        self,
        service_name: str,
        error: Exception,
        context: dict[str, object],
    ) -> object:
        """Return the last cached response for *service_name*.

        Parameters
        ----------
        service_name:
            Key used to look up the cached value.
        error:
            Logged at DEBUG level; otherwise ignored.
        context:
            Ignored.

        Returns
        -------
        object
            The most recently cached response.

        Raises
        ------
        KeyError
            If no cached value exists for *service_name*.
        """
        if service_name not in self._cache:
            raise KeyError(
                f"CacheFallbackHandler: no cached value for service={service_name!r}."
            )
        logger.debug(
            "CacheFallbackHandler: service=%r — returning cached response.",
            service_name,
        )
        return self._cache[service_name]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class FallbackRegistry:
    """Registry mapping service names to :class:`FallbackHandler` instances.

    A single handler is registered per service name.  Re-registering a
    name replaces the previous handler.

    Example
    -------
    ::

        registry = FallbackRegistry()
        registry.register("payments-api", StaticFallbackHandler({"error": "unavailable"}))

        result = registry.execute_fallback(
            "payments-api",
            error=CircuitBreakerOpenError("payments-api"),
        )
    """

    def __init__(self) -> None:
        self._handlers: dict[str, FallbackHandler] = {}

    def register(self, service_name: str, handler: FallbackHandler) -> None:
        """Register a fallback handler for a service.

        Parameters
        ----------
        service_name:
            Unique identifier for the service.  Case-sensitive.
        handler:
            The :class:`FallbackHandler` instance to associate with
            *service_name*.  Replaces any previously registered handler.
        """
        if not service_name:
            raise ValueError("service_name must not be empty.")
        self._handlers[service_name] = handler
        logger.debug(
            "FallbackRegistry: registered handler %r for service=%r.",
            type(handler).__name__,
            service_name,
        )

    def get_handler(self, service_name: str) -> FallbackHandler | None:
        """Return the handler for *service_name*, or ``None`` if absent.

        Parameters
        ----------
        service_name:
            Service identifier to look up.

        Returns
        -------
        FallbackHandler | None
        """
        return self._handlers.get(service_name)

    def has_fallback(self, service_name: str) -> bool:
        """Return True if a handler is registered for *service_name*.

        Parameters
        ----------
        service_name:
            Service identifier to check.
        """
        return service_name in self._handlers

    def execute_fallback(
        self,
        service_name: str,
        error: Exception,
        context: dict[str, object] | None = None,
    ) -> object:
        """Look up the handler for *service_name* and invoke it.

        Parameters
        ----------
        service_name:
            Service identifier whose fallback should be invoked.
        error:
            The exception that triggered the fallback.
        context:
            Optional caller-supplied context passed to the handler.
            Defaults to an empty dict if ``None``.

        Returns
        -------
        object
            Whatever the handler returns.

        Raises
        ------
        KeyError
            If no handler is registered for *service_name*.
        """
        handler = self._handlers.get(service_name)
        if handler is None:
            raise KeyError(
                f"FallbackRegistry: no handler registered for service={service_name!r}."
            )
        resolved_context: dict[str, object] = context if context is not None else {}
        logger.info(
            "FallbackRegistry: invoking %r fallback for service=%r.",
            type(handler).__name__,
            service_name,
        )
        return handler.handle(service_name, error, resolved_context)

    def unregister(self, service_name: str) -> None:
        """Remove the handler for *service_name*.

        Parameters
        ----------
        service_name:
            Service identifier to remove.

        Raises
        ------
        KeyError
            If no handler is registered for *service_name*.
        """
        if service_name not in self._handlers:
            raise KeyError(
                f"FallbackRegistry: cannot unregister — no handler for service={service_name!r}."
            )
        del self._handlers[service_name]
        logger.debug("FallbackRegistry: unregistered handler for service=%r.", service_name)

    def registered_services(self) -> list[str]:
        """Return a sorted list of service names with registered handlers."""
        return sorted(self._handlers)

    def __repr__(self) -> str:
        services = list(self._handlers)
        return f"FallbackRegistry(services={services!r})"


__all__ = [
    "FallbackHandler",
    "StaticFallbackHandler",
    "CacheFallbackHandler",
    "FallbackRegistry",
]
