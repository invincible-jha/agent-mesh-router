"""Tests for fallback_registry module.

Covers:
- FallbackHandler ABC contract
- StaticFallbackHandler returns correct value in all scenarios
- CacheFallbackHandler cache population and hit/miss behaviour
- FallbackRegistry register, unregister, has_fallback, execute_fallback
- Error cases: missing handlers, empty service names, invalid registrations
- Context propagation
- repr() outputs
"""
from __future__ import annotations

from typing import Any

import pytest

from agent_mesh_router.resilience.fallback_registry import (
    CacheFallbackHandler,
    FallbackHandler,
    FallbackRegistry,
    StaticFallbackHandler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SpyHandler(FallbackHandler):
    """Records calls for assertion purposes."""

    def __init__(self, return_value: Any = "spy_result") -> None:
        self.calls: list[tuple[str, Exception, dict[str, Any]]] = []
        self._return_value = return_value

    def handle(
        self,
        service_name: str,
        error: Exception,
        context: dict[str, Any],
    ) -> Any:
        self.calls.append((service_name, error, dict(context)))
        return self._return_value


def _make_error(msg: str = "boom") -> Exception:
    return RuntimeError(msg)


# ---------------------------------------------------------------------------
# StaticFallbackHandler
# ---------------------------------------------------------------------------


class TestStaticFallbackHandler:
    def test_returns_static_value(self) -> None:
        handler = StaticFallbackHandler(default_value={"status": "degraded"})
        result = handler.handle("svc", _make_error(), {})
        assert result == {"status": "degraded"}

    def test_returns_none_when_none_configured(self) -> None:
        handler = StaticFallbackHandler(default_value=None)
        assert handler.handle("svc", _make_error(), {}) is None

    def test_returns_zero_when_zero_configured(self) -> None:
        handler = StaticFallbackHandler(default_value=0)
        assert handler.handle("svc", _make_error(), {}) == 0

    def test_returns_list(self) -> None:
        handler = StaticFallbackHandler(default_value=[])
        assert handler.handle("svc", _make_error(), {}) == []

    def test_default_value_property(self) -> None:
        handler = StaticFallbackHandler(default_value=42)
        assert handler.default_value == 42

    def test_service_name_ignored(self) -> None:
        handler = StaticFallbackHandler(default_value="ok")
        assert handler.handle("alpha", _make_error(), {}) == "ok"
        assert handler.handle("beta", _make_error(), {}) == "ok"

    def test_context_ignored(self) -> None:
        handler = StaticFallbackHandler(default_value=True)
        result = handler.handle("svc", _make_error(), {"user_id": "u1", "request_id": "r1"})
        assert result is True

    def test_error_ignored(self) -> None:
        handler = StaticFallbackHandler(default_value="safe")
        result = handler.handle("svc", ValueError("bad input"), {})
        assert result == "safe"


# ---------------------------------------------------------------------------
# CacheFallbackHandler
# ---------------------------------------------------------------------------


class TestCacheFallbackHandler:
    def test_cache_miss_raises_key_error(self) -> None:
        handler = CacheFallbackHandler()
        with pytest.raises(KeyError, match="no cached value for service"):
            handler.handle("missing-svc", _make_error(), {})

    def test_update_then_handle_returns_cached_value(self) -> None:
        handler = CacheFallbackHandler()
        handler.update_cache("payments", {"total": 0})
        result = handler.handle("payments", _make_error(), {})
        assert result == {"total": 0}

    def test_update_overwrites_previous_value(self) -> None:
        handler = CacheFallbackHandler()
        handler.update_cache("svc", "v1")
        handler.update_cache("svc", "v2")
        assert handler.handle("svc", _make_error(), {}) == "v2"

    def test_multiple_services_independent(self) -> None:
        handler = CacheFallbackHandler()
        handler.update_cache("a", 1)
        handler.update_cache("b", 2)
        assert handler.handle("a", _make_error(), {}) == 1
        assert handler.handle("b", _make_error(), {}) == 2

    def test_has_cached_value_true(self) -> None:
        handler = CacheFallbackHandler()
        handler.update_cache("svc", "x")
        assert handler.has_cached_value("svc") is True

    def test_has_cached_value_false(self) -> None:
        handler = CacheFallbackHandler()
        assert handler.has_cached_value("svc") is False

    def test_context_does_not_affect_result(self) -> None:
        handler = CacheFallbackHandler()
        handler.update_cache("svc", "cached")
        result = handler.handle("svc", _make_error(), {"k": "v"})
        assert result == "cached"

    def test_cache_none_value(self) -> None:
        handler = CacheFallbackHandler()
        handler.update_cache("svc", None)
        assert handler.handle("svc", _make_error(), {}) is None
        assert handler.has_cached_value("svc") is True


# ---------------------------------------------------------------------------
# FallbackRegistry
# ---------------------------------------------------------------------------


class TestFallbackRegistry:
    def test_register_and_has_fallback(self) -> None:
        registry = FallbackRegistry()
        registry.register("svc", StaticFallbackHandler("ok"))
        assert registry.has_fallback("svc") is True

    def test_has_fallback_false_for_unknown(self) -> None:
        registry = FallbackRegistry()
        assert registry.has_fallback("unknown") is False

    def test_get_handler_returns_handler(self) -> None:
        registry = FallbackRegistry()
        handler = StaticFallbackHandler("val")
        registry.register("svc", handler)
        assert registry.get_handler("svc") is handler

    def test_get_handler_returns_none_for_unknown(self) -> None:
        registry = FallbackRegistry()
        assert registry.get_handler("unknown") is None

    def test_execute_fallback_dispatches_to_handler(self) -> None:
        registry = FallbackRegistry()
        spy = _SpyHandler(return_value="fallback!")
        registry.register("svc", spy)
        error = _make_error()
        result = registry.execute_fallback("svc", error, context={"k": "v"})
        assert result == "fallback!"
        assert len(spy.calls) == 1
        name, exc, ctx = spy.calls[0]
        assert name == "svc"
        assert exc is error
        assert ctx == {"k": "v"}

    def test_execute_fallback_defaults_context_to_empty_dict(self) -> None:
        registry = FallbackRegistry()
        spy = _SpyHandler()
        registry.register("svc", spy)
        registry.execute_fallback("svc", _make_error())
        _, _, ctx = spy.calls[0]
        assert ctx == {}

    def test_execute_fallback_raises_key_error_when_no_handler(self) -> None:
        registry = FallbackRegistry()
        with pytest.raises(KeyError, match="no handler registered"):
            registry.execute_fallback("missing", _make_error())

    def test_register_empty_service_name_raises(self) -> None:
        registry = FallbackRegistry()
        with pytest.raises(ValueError, match="service_name must not be empty"):
            registry.register("", StaticFallbackHandler("x"))

    def test_register_replaces_existing_handler(self) -> None:
        registry = FallbackRegistry()
        registry.register("svc", StaticFallbackHandler("first"))
        registry.register("svc", StaticFallbackHandler("second"))
        result = registry.execute_fallback("svc", _make_error())
        assert result == "second"

    def test_unregister_removes_handler(self) -> None:
        registry = FallbackRegistry()
        registry.register("svc", StaticFallbackHandler("x"))
        registry.unregister("svc")
        assert registry.has_fallback("svc") is False

    def test_unregister_raises_key_error_when_not_registered(self) -> None:
        registry = FallbackRegistry()
        with pytest.raises(KeyError, match="cannot unregister"):
            registry.unregister("ghost")

    def test_registered_services_sorted(self) -> None:
        registry = FallbackRegistry()
        registry.register("zebra", StaticFallbackHandler(0))
        registry.register("alpha", StaticFallbackHandler(0))
        registry.register("mango", StaticFallbackHandler(0))
        assert registry.registered_services() == ["alpha", "mango", "zebra"]

    def test_registered_services_empty_initially(self) -> None:
        registry = FallbackRegistry()
        assert registry.registered_services() == []

    def test_repr_contains_service_names(self) -> None:
        registry = FallbackRegistry()
        registry.register("svc-a", StaticFallbackHandler(0))
        representation = repr(registry)
        assert "svc-a" in representation

    def test_cache_handler_integrated_with_registry(self) -> None:
        registry = FallbackRegistry()
        cache_handler = CacheFallbackHandler()
        cache_handler.update_cache("inventory", {"items": [1, 2, 3]})
        registry.register("inventory", cache_handler)
        result = registry.execute_fallback("inventory", _make_error())
        assert result == {"items": [1, 2, 3]}

    def test_multiple_services_each_get_own_handler(self) -> None:
        registry = FallbackRegistry()
        registry.register("svc-a", StaticFallbackHandler("a-result"))
        registry.register("svc-b", StaticFallbackHandler("b-result"))
        assert registry.execute_fallback("svc-a", _make_error()) == "a-result"
        assert registry.execute_fallback("svc-b", _make_error()) == "b-result"
