"""Plugin registry for agent-mesh-router.

Provides a decorator-based registration system for plugins.
Third-party implementations register via this system by declaring
entry-points in their own ``pyproject.toml`` under the
"agent_mesh_router.plugins" group.

Example
-------
Define a base class and registry::

    from abc import ABC, abstractmethod
    from agent_mesh_router.plugins.registry import PluginRegistry

    class BaseProcessor(ABC):
        @abstractmethod
        def process(self, data: bytes) -> bytes: ...

    processor_registry: PluginRegistry[BaseProcessor] = PluginRegistry(
        BaseProcessor, "processors"
    )

Register a plugin with the decorator::

    @processor_registry.register("my-processor")
    class MyProcessor(BaseProcessor):
        def process(self, data: bytes) -> bytes:
            return data.upper()

Load all installed plugins via entry-points::

    processor_registry.load_entrypoints("agent_mesh_router.plugins")

Retrieve a plugin by name::

    cls = processor_registry.get("my-processor")
    instance = cls()
"""
from __future__ import annotations

import importlib.metadata
import logging
from abc import ABC
from collections.abc import Callable
from typing import Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=ABC)


class PluginNotFoundError(KeyError):
    """Raised when a requested plugin name is not in the registry."""

    def __init__(self, name: str, registry_name: str) -> None:
        self.plugin_name = name
        self.registry_name = registry_name
        super().__init__(
            f"Plugin {name!r} is not registered in the {registry_name!r} registry. "
            f"Available plugins: {name!r} was not found. "
            "Check that the package is installed and its entry-points are declared."
        )


class PluginAlreadyRegisteredError(ValueError):
    """Raised when attempting to register a name that already exists."""

    def __init__(self, name: str, registry_name: str) -> None:
        self.plugin_name = name
        self.registry_name = registry_name
        super().__init__(
            f"Plugin {name!r} is already registered in the {registry_name!r} registry. "
            "Use a unique name or explicitly deregister the existing entry first."
        )


class PluginRegistry(Generic[T]):
    """Type-safe registry for plugin implementations.

    Plugins are registered either via the ``@register`` decorator at
    import time, or lazily via ``load_entrypoints`` for installed packages.

    Parameters
    ----------
    base_class:
        The abstract base class all plugins must subclass.
    name:
        A human-readable name for this registry (used in error messages).
    """

    def __init__(self, base_class: type[T], name: str) -> None:
        self._base_class = base_class
        self._name = name
        self._plugins: dict[str, type[T]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Return a class decorator that registers the decorated class.

        Parameters
        ----------
        name:
            The unique string key for this plugin.

        Returns
        -------
        Callable[[type[T]], type[T]]
            A decorator that registers the class and returns it unchanged,
            allowing it to remain usable as a normal class.

        Raises
        ------
        PluginAlreadyRegisteredError
            If ``name`` is already in use in this registry.
        TypeError
            If the decorated class does not subclass ``base_class``.

        Example
        -------
        ::

            @registry.register("my-plugin")
            class MyPlugin(BasePlugin):
                ...
        """

        def decorator(cls: type[T]) -> type[T]:
            if name in self._plugins:
                raise PluginAlreadyRegisteredError(name, self._name)
            if not (isinstance(cls, type) and issubclass(cls, self._base_class)):
                raise TypeError(
                    f"Cannot register {cls!r} under {name!r}: "
                    f"it must be a subclass of {self._base_class.__name__}."
                )
            self._plugins[name] = cls
            logger.debug(
                "Registered plugin %r -> %s in registry %r",
                name,
                cls.__qualname__,
                self._name,
            )
            return cls

        return decorator

    def register_class(self, name: str, cls: type[T]) -> None:
        """Register a class directly without using the decorator syntax.

        This is useful for programmatic registration, such as inside
        ``load_entrypoints``.

        Parameters
        ----------
        name:
            The unique string key for this plugin.
        cls:
            The class to register. Must subclass ``base_class``.

        Raises
        ------
        PluginAlreadyRegisteredError
            If ``name`` is already registered.
        TypeError
            If ``cls`` is not a subclass of ``base_class``.
        """
        if name in self._plugins:
            raise PluginAlreadyRegisteredError(name, self._name)
        if not (isinstance(cls, type) and issubclass(cls, self._base_class)):
            raise TypeError(
                f"Cannot register {cls!r} under {name!r}: "
                f"it must be a subclass of {self._base_class.__name__}."
            )
        self._plugins[name] = cls
        logger.debug(
            "Registered plugin %r -> %s in registry %r",
            name,
            cls.__qualname__,
            self._name,
        )

    def deregister(self, name: str) -> None:
        """Remove a plugin from the registry.

        Parameters
        ----------
        name:
            The key previously passed to ``register``.

        Raises
        ------
        PluginNotFoundError
            If ``name`` is not currently registered.
        """
        if name not in self._plugins:
            raise PluginNotFoundError(name, self._name)
        del self._plugins[name]
        logger.debug("Deregistered plugin %r from registry %r", name, self._name)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> type[T]:
        """Return the class registered under ``name``.

        Parameters
        ----------
        name:
            The key used when registering the plugin.

        Returns
        -------
        type[T]
            The registered class (not an instance).

        Raises
        ------
        PluginNotFoundError
            If no plugin is registered under ``name``.
        """
        try:
            return self._plugins[name]
        except KeyError:
            raise PluginNotFoundError(name, self._name) from None

    def list_plugins(self) -> list[str]:
        """Return a sorted list of all registered plugin names.

        Returns
        -------
        list[str]
            Plugin names in alphabetical order.
        """
        return sorted(self._plugins)

    def __contains__(self, name: object) -> bool:
        """Support ``"my-plugin" in registry`` membership test."""
        return name in self._plugins

    def __len__(self) -> int:
        """Return the number of registered plugins."""
        return len(self._plugins)

    def __repr__(self) -> str:
        return (
            f"PluginRegistry(name={self._name!r}, "
            f"base_class={self._base_class.__name__}, "
            f"plugins={self.list_plugins()})"
        )

    # ------------------------------------------------------------------
    # Entry-point loading
    # ------------------------------------------------------------------

    def load_entrypoints(self, group: str) -> None:
        """Discover and register plugins declared as package entry-points.

        Iterates over all installed distributions that declare entry-points
        in ``group``. Each entry-point value is imported and registered
        under the entry-point name.

        Plugins that are already registered (e.g., from a previous call)
        are skipped with a debug-level log entry rather than raising an
        error. This makes repeated calls to ``load_entrypoints`` idempotent.

        Parameters
        ----------
        group:
            The entry-point group name, e.g. "agent_mesh_router.plugins".

        Example
        -------
        In a downstream package's ``pyproject.toml``::

            [agent_mesh_router.plugins]
            my-processor = "my_package.processors:MyProcessor"

        Then at runtime::

            registry.load_entrypoints("agent_mesh_router.plugins")
        """
        entry_points = importlib.metadata.entry_points(group=group)
        for ep in entry_points:
            if ep.name in self._plugins:
                logger.debug(
                    "Entry-point %r already registered in %r; skipping.",
                    ep.name,
                    self._name,
                )
                continue
            try:
                cls = ep.load()
            except Exception:
                logger.exception(
                    "Failed to load entry-point %r from group %r; skipping.",
                    ep.name,
                    group,
                )
                continue
            try:
                self.register_class(ep.name, cls)
            except (PluginAlreadyRegisteredError, TypeError):
                logger.warning(
                    "Entry-point %r loaded but could not be registered "
                    "in registry %r; skipping.",
                    ep.name,
                    self._name,
                )
