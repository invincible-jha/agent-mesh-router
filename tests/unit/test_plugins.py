"""Tests for PluginRegistry."""
from __future__ import annotations

from abc import ABC, abstractmethod
from unittest.mock import MagicMock, patch

import pytest

from agent_mesh_router.plugins.registry import (
    PluginAlreadyRegisteredError,
    PluginNotFoundError,
    PluginRegistry,
)


# ---------------------------------------------------------------------------
# Minimal abstract base for testing
# ---------------------------------------------------------------------------


class BaseWidget(ABC):
    @abstractmethod
    def render(self) -> str: ...


class ConcreteWidget(BaseWidget):
    def render(self) -> str:
        return "widget"


class AnotherWidget(BaseWidget):
    def render(self) -> str:
        return "another"


# ---------------------------------------------------------------------------
# PluginNotFoundError / PluginAlreadyRegisteredError
# ---------------------------------------------------------------------------


class TestPluginErrors:
    def test_not_found_error_contains_name(self) -> None:
        err = PluginNotFoundError("missing", "widgets")
        assert "missing" in str(err)
        assert err.plugin_name == "missing"
        assert err.registry_name == "widgets"

    def test_already_registered_error_contains_name(self) -> None:
        err = PluginAlreadyRegisteredError("dup", "widgets")
        assert "dup" in str(err)
        assert err.plugin_name == "dup"
        assert err.registry_name == "widgets"


# ---------------------------------------------------------------------------
# PluginRegistry — registration via decorator
# ---------------------------------------------------------------------------


class TestPluginRegistryDecorator:
    def _registry(self) -> PluginRegistry[BaseWidget]:
        return PluginRegistry(BaseWidget, "widgets")

    def test_register_and_get(self) -> None:
        registry = self._registry()

        @registry.register("my-widget")
        class MyW(BaseWidget):
            def render(self) -> str:
                return "my"

        cls = registry.get("my-widget")
        assert cls is MyW

    def test_register_duplicate_raises(self) -> None:
        registry = self._registry()
        registry.register_class("existing", ConcreteWidget)
        with pytest.raises(PluginAlreadyRegisteredError):
            registry.register_class("existing", AnotherWidget)

    def test_register_wrong_type_raises(self) -> None:
        registry = self._registry()

        class NotAWidget:
            pass

        with pytest.raises(TypeError):
            registry.register_class("bad", NotAWidget)  # type: ignore[arg-type]

    def test_register_decorator_raises_already_registered(self) -> None:
        registry = self._registry()
        registry.register_class("taken", ConcreteWidget)

        with pytest.raises(PluginAlreadyRegisteredError):
            @registry.register("taken")
            class DupWidget(BaseWidget):
                def render(self) -> str:
                    return "dup"

    def test_get_nonexistent_raises(self) -> None:
        registry = self._registry()
        with pytest.raises(PluginNotFoundError):
            registry.get("missing")

    def test_contains(self) -> None:
        registry = self._registry()
        registry.register_class("present", ConcreteWidget)
        assert "present" in registry
        assert "absent" not in registry

    def test_len(self) -> None:
        registry = self._registry()
        assert len(registry) == 0
        registry.register_class("a", ConcreteWidget)
        assert len(registry) == 1

    def test_list_plugins_sorted(self) -> None:
        registry = self._registry()
        registry.register_class("b-plugin", ConcreteWidget)
        registry.register_class("a-plugin", AnotherWidget)
        names = registry.list_plugins()
        assert names == sorted(names)
        assert "a-plugin" in names
        assert "b-plugin" in names

    def test_deregister_removes_plugin(self) -> None:
        registry = self._registry()
        registry.register_class("to-remove", ConcreteWidget)
        registry.deregister("to-remove")
        assert "to-remove" not in registry

    def test_deregister_nonexistent_raises(self) -> None:
        registry = self._registry()
        with pytest.raises(PluginNotFoundError):
            registry.deregister("ghost")

    def test_repr_contains_registry_name(self) -> None:
        registry = self._registry()
        assert "widgets" in repr(registry)

    def test_register_class_validates_subclass(self) -> None:
        registry = self._registry()

        class Unrelated(ABC):
            @abstractmethod
            def something(self) -> None: ...

        with pytest.raises(TypeError):
            registry.register_class("bad", Unrelated)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# PluginRegistry — load_entrypoints
# ---------------------------------------------------------------------------


class TestLoadEntrypoints:
    def _registry(self) -> PluginRegistry[BaseWidget]:
        return PluginRegistry(BaseWidget, "widgets")

    def test_load_entrypoints_registers_valid_class(self) -> None:
        registry = self._registry()

        mock_ep = MagicMock()
        mock_ep.name = "ep-widget"
        mock_ep.load.return_value = ConcreteWidget

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            registry.load_entrypoints("agent_mesh_router.plugins")

        assert "ep-widget" in registry
        assert registry.get("ep-widget") is ConcreteWidget

    def test_load_entrypoints_skips_already_registered(self) -> None:
        registry = self._registry()
        registry.register_class("ep-widget", ConcreteWidget)

        mock_ep = MagicMock()
        mock_ep.name = "ep-widget"
        mock_ep.load.return_value = AnotherWidget

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            registry.load_entrypoints("agent_mesh_router.plugins")

        # Should remain the originally registered class.
        assert registry.get("ep-widget") is ConcreteWidget

    def test_load_entrypoints_handles_load_failure(self) -> None:
        registry = self._registry()

        mock_ep = MagicMock()
        mock_ep.name = "bad-ep"
        mock_ep.load.side_effect = ImportError("no module")

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            registry.load_entrypoints("agent_mesh_router.plugins")

        assert "bad-ep" not in registry

    def test_load_entrypoints_handles_type_error_registration(self) -> None:
        registry = self._registry()

        class NotAWidget:
            pass

        mock_ep = MagicMock()
        mock_ep.name = "type-error-ep"
        mock_ep.load.return_value = NotAWidget

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            registry.load_entrypoints("agent_mesh_router.plugins")

        assert "type-error-ep" not in registry

    def test_load_entrypoints_idempotent_on_repeated_calls(self) -> None:
        registry = self._registry()

        mock_ep = MagicMock()
        mock_ep.name = "idempotent"
        mock_ep.load.return_value = ConcreteWidget

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            registry.load_entrypoints("group")
            registry.load_entrypoints("group")  # Second call should not raise.

        assert "idempotent" in registry

    def test_empty_entrypoints_no_error(self) -> None:
        registry = self._registry()
        with patch("importlib.metadata.entry_points", return_value=[]):
            registry.load_entrypoints("empty.group")
        assert len(registry) == 0
