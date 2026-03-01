# Examples

| # | Example | Description |
|---|---------|-------------|
| 01 | [Quickstart](01_quickstart.py) | Register agents, route a message, run a sequential workflow |
| 02 | [Routing Strategies](02_routing_strategies.py) | RoundRobin, LeastLoaded, CapabilityMatch, and Composite routing |
| 03 | [Workflow Orchestration](03_workflow_orchestration.py) | Sequential, Parallel, and Competitive workflow executors |
| 04 | [Fleet Registry](04_fleet_registry.py) | Agent node registration, health monitoring, and load balancing |
| 05 | [Resilience](05_resilience.py) | Circuit breaker state management and retry with backoff |
| 06 | [Pub/Sub Broker](06_pubsub_broker.py) | Topic-based pub/sub and priority message queue |
| 07 | [LangChain Routing](07_langchain_routing.py) | Route tasks to LangChain runnables via the mesh router |

## Running the examples

```bash
pip install agent-mesh-router
python examples/01_quickstart.py
```

For framework integrations:

```bash
pip install langchain   # for example 07
```
