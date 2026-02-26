# Architecture — agent-mesh-router

## Overview

Multi-agent communication, task routing, and workflow orchestration

This document describes the high-level architecture of agent-mesh-router
and the design decisions behind it.

## Component Map

```
agent-mesh-router/
  src/agent_mesh_router/
    core/        # Domain logic, models, protocols
    plugins/     # Plugin registry and base classes
    cli/         # Click CLI application
```

## Plugin System

agent-mesh-router uses a decorator-based plugin registry backed by
``importlib.metadata`` entry-points. This allows third-party packages
(including the AgentMesh enterprise edition) to extend the system
without modifying the core.

### Registration at import time

```python
from agent_mesh_router.plugins.registry import PluginRegistry
from agent_mesh_router.core import BaseProcessor  # example base class

processor_registry: PluginRegistry[BaseProcessor] = PluginRegistry(
    BaseProcessor, "processors"
)

@processor_registry.register("my-processor")
class MyProcessor(BaseProcessor):
    ...
```

### Registration via entry-points

Downstream packages declare plugins in ``pyproject.toml``:

```toml
[agent_mesh_router.plugins]
my-processor = "my_package:MyProcessor"
```

Then load them at startup:

```python
processor_registry.load_entrypoints("agent_mesh_router.plugins")
```

## Design Principles

- **Dependency injection**: services receive dependencies as constructor
  arguments rather than reaching for globals.
- **Pydantic v2 at boundaries**: all data entering or leaving the system
  is validated via Pydantic models.
- **Async-first**: I/O-bound operations use ``async``/``await``.
- **No hidden globals**: avoid module-level singletons that complicate
  testing and concurrent use.

## Extension Points

| Extension Point | Mechanism |
|----------------|-----------|
| Custom processors | ``PluginRegistry`` entry-points |
| Custom CLI commands | ``click`` group plugins |
| Configuration | Pydantic ``BaseSettings`` |

## Future Work

- [ ] Async streaming support
- [ ] OpenTelemetry tracing
- [ ] gRPC transport option
