"""agent-mesh-router — Multi-agent communication, task routing, and workflow orchestration.

Public API
----------
The stable public surface is everything exported from this module.
Anything inside submodules not re-exported here is considered private
and may change without notice.

Packages
--------
messages
    Core message types, envelope definition, and serialization.
routing
    Routing table, strategies, and route resolver.
broker
    Async message broker, priority queue, dead-letter queue, pub/sub.
fleet
    Agent node registry, health monitoring, and load balancing.
workflows
    Sequential, parallel, hierarchical, competitive, and consensus executors.
backpressure
    Queue depth monitoring and adaptive token-bucket throttling.
adapters
    HTTP, gRPC (stub), Redis, and Kafka transport adapters.
middleware
    Tracing middleware for distributed trace context injection.
plugins
    Plugin registry with entry-point loading.
cli
    Command-line interface.

Example
-------
::

    import agent_mesh_router
    print(agent_mesh_router.__version__)
    # '0.1.0'

    from agent_mesh_router import (
        MessageEnvelope,
        MessageType,
        Priority,
        AsyncMessageBroker,
        SequentialWorkflow,
        WorkflowStep,
        FleetRegistry,
        AgentNode,
        BackpressureMonitor,
        AdaptiveThrottle,
        TracingMiddleware,
    )
"""
from __future__ import annotations

__version__: str = "0.1.0"

# ------------------------------------------------------------------
# messages
# ------------------------------------------------------------------
from agent_mesh_router.messages.types import (
    HandoffMetrics,
    MessageType,
    Priority,
)
from agent_mesh_router.messages.envelope import MessageEnvelope
from agent_mesh_router.messages.serializer import MessageSerializer

# ------------------------------------------------------------------
# routing
# ------------------------------------------------------------------
from agent_mesh_router.routing.table import (
    AgentRecord,
    AgentAlreadyRegisteredError,
    AgentNotFoundError,
    RoutingTable,
)
from agent_mesh_router.routing.strategy import (
    CapabilityMatchRouter,
    CompositeRouter,
    CostAwareRouter,
    LeastLoadedRouter,
    RoundRobinRouter,
    RoutingStrategy,
)
from agent_mesh_router.routing.resolver import RouteResolver

# ------------------------------------------------------------------
# broker
# ------------------------------------------------------------------
from agent_mesh_router.broker.queue import (
    PriorityMessageQueue,
    QueueFullError,
)
from agent_mesh_router.broker.dead_letter import (
    DeadLetterEntry,
    DeadLetterQueue,
)
from agent_mesh_router.broker.message_broker import AsyncMessageBroker
from agent_mesh_router.broker.pubsub import TopicManager, TopicSubscription

# ------------------------------------------------------------------
# fleet
# ------------------------------------------------------------------
from agent_mesh_router.fleet.registry import (
    AgentNode,
    AgentNodeNotFoundError,
    AgentStatus,
    FleetRegistry,
)
from agent_mesh_router.fleet.health import HealthMonitor
from agent_mesh_router.fleet.load_balancer import LoadBalancer, Strategy

# ------------------------------------------------------------------
# workflows
# ------------------------------------------------------------------
from agent_mesh_router.workflows.base import (
    StepResult,
    WorkflowExecutor,
    WorkflowResult,
    WorkflowStatus,
    WorkflowStep,
)
from agent_mesh_router.workflows.sequential import SequentialWorkflow
from agent_mesh_router.workflows.parallel import ParallelWorkflow
from agent_mesh_router.workflows.hierarchical import HierarchicalWorkflow
from agent_mesh_router.workflows.competitive import CompetitiveWorkflow
from agent_mesh_router.workflows.consensus import ConsensusWorkflow

# ------------------------------------------------------------------
# backpressure
# ------------------------------------------------------------------
from agent_mesh_router.backpressure.monitor import (
    BackpressureMonitor,
    CRITICAL_THRESHOLD,
    QueueMetrics,
    WARN_THRESHOLD,
)
from agent_mesh_router.backpressure.throttle import AdaptiveThrottle

# ------------------------------------------------------------------
# adapters
# ------------------------------------------------------------------
from agent_mesh_router.adapters.http import HttpTransport, HttpTransportError
from agent_mesh_router.adapters.grpc_adapter import (
    GrpcTransport,
    GrpcTransportError,
)

# Optional adapters — only imported when deps are present
try:
    from agent_mesh_router.adapters.redis_adapter import (
        RedisTransport,
        RedisTransportError,
    )
except ImportError:
    pass  # redis not installed

try:
    from agent_mesh_router.adapters.kafka_adapter import (
        KafkaTransport,
        KafkaTransportError,
    )
except ImportError:
    pass  # aiokafka not installed

# ------------------------------------------------------------------
# middleware
# ------------------------------------------------------------------
from agent_mesh_router.middleware.tracing import TracingMiddleware

# ------------------------------------------------------------------
# plugins
# ------------------------------------------------------------------
from agent_mesh_router.plugins.registry import (
    PluginAlreadyRegisteredError,
    PluginNotFoundError,
    PluginRegistry,
)

# ------------------------------------------------------------------
# resilience
# ------------------------------------------------------------------
from agent_mesh_router.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitState,
)
from agent_mesh_router.resilience.retry import (
    RetryConfig,
    RetryExhaustedError,
    RetryPolicy,
)
from agent_mesh_router.resilience.load_balancer import (
    LoadBalancer as ResilienceLoadBalancer,
    LoadBalancerStrategy,
    NoHealthyEndpointError,
    ServiceEndpoint,
)
from agent_mesh_router.resilience.health_check import (
    HealthCheckConfig,
    HealthCheckResult,
    HealthChecker,
    HealthStatus,
)

__all__: list[str] = [
    "__version__",
    # messages
    "HandoffMetrics",
    "MessageEnvelope",
    "MessageSerializer",
    "MessageType",
    "Priority",
    # routing
    "AgentAlreadyRegisteredError",
    "AgentNotFoundError",
    "AgentRecord",
    "CapabilityMatchRouter",
    "CompositeRouter",
    "CostAwareRouter",
    "LeastLoadedRouter",
    "RouteResolver",
    "RoutingStrategy",
    "RoutingTable",
    "RoundRobinRouter",
    # broker
    "AsyncMessageBroker",
    "DeadLetterEntry",
    "DeadLetterQueue",
    "PriorityMessageQueue",
    "QueueFullError",
    "TopicManager",
    "TopicSubscription",
    # fleet
    "AgentNode",
    "AgentNodeNotFoundError",
    "AgentStatus",
    "FleetRegistry",
    "HealthMonitor",
    "LoadBalancer",
    "Strategy",
    # workflows
    "CompetitiveWorkflow",
    "ConsensusWorkflow",
    "HierarchicalWorkflow",
    "ParallelWorkflow",
    "SequentialWorkflow",
    "StepResult",
    "WorkflowExecutor",
    "WorkflowResult",
    "WorkflowStatus",
    "WorkflowStep",
    # backpressure
    "AdaptiveThrottle",
    "BackpressureMonitor",
    "CRITICAL_THRESHOLD",
    "QueueMetrics",
    "WARN_THRESHOLD",
    # adapters
    "GrpcTransport",
    "GrpcTransportError",
    "HttpTransport",
    "HttpTransportError",
    # middleware
    "TracingMiddleware",
    # plugins
    "PluginAlreadyRegisteredError",
    "PluginNotFoundError",
    "PluginRegistry",
    # resilience
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerOpenError",
    "CircuitState",
    "RetryConfig",
    "RetryExhaustedError",
    "RetryPolicy",
    "ResilienceLoadBalancer",
    "LoadBalancerStrategy",
    "NoHealthyEndpointError",
    "ServiceEndpoint",
    "HealthCheckConfig",
    "HealthCheckResult",
    "HealthChecker",
    "HealthStatus",
]
