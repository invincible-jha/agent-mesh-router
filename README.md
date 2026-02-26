# agent-mesh-router

Multi-agent communication, task routing, and workflow orchestration

[![CI](https://github.com/aumos-ai/agent-mesh-router/actions/workflows/ci.yaml/badge.svg)](https://github.com/aumos-ai/agent-mesh-router/actions/workflows/ci.yaml)
[![PyPI version](https://img.shields.io/pypi/v/agent-mesh-router.svg)](https://pypi.org/project/agent-mesh-router/)
[![Python versions](https://img.shields.io/pypi/pyversions/agent-mesh-router.svg)](https://pypi.org/project/agent-mesh-router/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Part of the [AumOS](https://github.com/aumos-ai) open-source agent infrastructure portfolio.

---

## Features

- Five routing strategies — `CapabilityMatchRouter`, `CostAwareRouter`, `RoundRobinRouter`, `LeastLoadedRouter`, and `CompositeRouter` (weighted combination) — all implementing a common `RoutingStrategy` ABC
- Async `MessageBroker` with in-memory pub/sub, dead-letter queue support, and backpressure monitoring that throttles producers when consumers fall behind
- Five workflow patterns: `SequentialWorkflow`, `ParallelWorkflow`, `HierarchicalWorkflow`, `CompetitiveWorkflow` (first-result wins), and `ConsensusWorkflow` (majority agreement)
- Fleet registry with health checking and a load balancer that routes only to healthy agents based on current load metrics
- Transport adapters for HTTP, gRPC, Redis pub/sub, and Kafka so the same router can operate across different message infrastructure
- `MessageEnvelope` with typed priority levels, routing metadata, TTL, and checksum validation for tamper detection
- OpenTelemetry tracing middleware that adds trace context propagation to every routed message

## Quick Start

Install from PyPI:

```bash
pip install agent-mesh-router
```

Verify the installation:

```bash
agent-mesh-router version
```

Basic usage:

```python
import agent_mesh_router

# See examples/01_quickstart.py for a working example
```

## Documentation

- [Architecture](docs/architecture.md)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)
- [Examples](examples/README.md)

## Enterprise Upgrade

The open-source edition provides the core foundation. For production
deployments requiring SLA-backed support, advanced integrations, and the full
AgentMesh platform, see [docs/UPGRADE_TO_AgentMesh.md](docs/UPGRADE_TO_AgentMesh.md).

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md)
before opening a pull request.

## License

Apache 2.0 — see [LICENSE](LICENSE) for full terms.

---

Part of [AumOS](https://github.com/aumos-ai) — open-source agent infrastructure.
