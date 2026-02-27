# Using agent-mesh-router Alongside CrewAI

agent-mesh-router and CrewAI solve different problems at different layers of the agent stack.
CrewAI is a framework for defining *what* agents do — their roles, goals, tools, and task
sequences. agent-mesh-router handles *how* messages move between agents at production scale —
routing, circuit breakers, backpressure, protocol adapters, and consensus workflows.

This guide shows how to use both together, with CrewAI handling agent definition and
agent-mesh-router handling production routing infrastructure.

---

## Feature Comparison

| Capability | CrewAI | agent-mesh-router |
|---|---|---|
| Agent role and goal definition | Yes — `Agent` with `role`, `goal`, `backstory` | Not in scope — bring your own agents |
| Sequential task execution | Yes — `Process.sequential` | Yes — `SequentialWorkflow` |
| Hierarchical task execution | Yes — `Process.hierarchical` | Yes — `HierarchicalWorkflow` |
| Parallel execution | No (sequential by default) | Yes — `ParallelWorkflow` |
| Competitive execution (first-wins) | No | Yes — `CompetitiveWorkflow` |
| Consensus / voting | No | Yes — `ConsensusWorkflow` |
| Protocol-agnostic transport | HTTP only | HTTP, gRPC, Redis pub/sub, Kafka |
| Circuit breakers | No | Yes — `CircuitBreaker` with CLOSED/OPEN/HALF-OPEN state machine |
| Backpressure / queue depth monitoring | No | Yes — `BackpressureMonitor` + `AdaptiveThrottle` |
| Retry with exponential backoff | Limited | Yes — `RetryPolicy` with configurable backoff |
| Dead-letter queue | No | Yes — `DeadLetterQueue` |
| Fleet registry and health monitoring | No | Yes — `FleetRegistry`, `HealthMonitor` |
| Capability-based routing | No | Yes — `CapabilityMatchRouter` |
| Cost-aware routing | No | Yes — `CostAwareRouter` |
| Pub/sub messaging | No | Yes — `TopicManager`, `TopicSubscription` |
| Priority queue | No | Yes — `PriorityMessageQueue` |

---

## Installation

```bash
pip install agent-mesh-router crewai

# With Redis or Kafka transport adapters
pip install "agent-mesh-router[redis]"
pip install "agent-mesh-router[kafka]"
```

---

## Pattern 1 — CrewAI for Agent Definition, agent-mesh-router for Routing

Define your agents and tasks in CrewAI as normal. Replace the crew's built-in execution with
agent-mesh-router's workflow engine so you gain circuit breakers and backpressure.

**Before (CrewAI only — direct crew execution):**

```python
from crewai import Agent, Task, Crew, Process

researcher = Agent(
    role="Research Analyst",
    goal="Find accurate information on the given topic",
    backstory="Expert at web research and data synthesis",
    verbose=True,
)
writer = Agent(
    role="Content Writer",
    goal="Transform research into clear written content",
    backstory="Skilled at explaining complex topics simply",
    verbose=True,
)

research_task = Task(
    description="Research the latest developments in quantum computing",
    expected_output="A bullet-point summary of recent breakthroughs",
    agent=researcher,
)
writing_task = Task(
    description="Write a blog post based on the research",
    expected_output="A 500-word blog post",
    agent=writer,
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,
)
result = crew.kickoff()
```

**After (CrewAI + agent-mesh-router — circuit-broken sequential workflow):**

```python
import asyncio
from crewai import Agent, Task

from agent_mesh_router import (
    SequentialWorkflow,
    WorkflowStep,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    FleetRegistry,
    AgentNode,
    AgentStatus,
)

# Define CrewAI agents as before
researcher = Agent(
    role="Research Analyst",
    goal="Find accurate information on the given topic",
    backstory="Expert at web research and data synthesis",
)
writer = Agent(
    role="Content Writer",
    goal="Transform research into clear written content",
    backstory="Skilled at explaining complex topics simply",
)

# Register agents in the fleet registry
registry = FleetRegistry()
registry.register(AgentNode(
    agent_id="researcher",
    capabilities={"research", "web-search"},
    status=AgentStatus.HEALTHY,
))
registry.register(AgentNode(
    agent_id="writer",
    capabilities={"writing", "summarization"},
    status=AgentStatus.HEALTHY,
))

# Wrap each CrewAI agent call in a circuit breaker
research_cb = CircuitBreaker(
    config=CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout_seconds=30.0,
        half_open_max_calls=2,
        success_threshold=2,
    ),
    name="researcher-cb",
)
writer_cb = CircuitBreaker(
    config=CircuitBreakerConfig(failure_threshold=3, recovery_timeout_seconds=30.0),
    name="writer-cb",
)

# Define an async executor that calls the CrewAI agent
async def execute_crewai_step(step: WorkflowStep) -> dict[str, object]:
    if step.agent_id == "researcher":
        task = Task(
            description=step.action,
            expected_output="Research findings",
            agent=researcher,
        )
        try:
            result_text = research_cb.execute(lambda: researcher.execute_task(task))
        except CircuitBreakerOpenError:
            return {"error": "researcher circuit open", "output": None}
        return {"output": result_text, "agent_id": "researcher"}

    elif step.agent_id == "writer":
        task = Task(
            description=step.action,
            expected_output="Written content",
            agent=writer,
        )
        try:
            result_text = writer_cb.execute(lambda: writer.execute_task(task))
        except CircuitBreakerOpenError:
            return {"error": "writer circuit open", "output": None}
        return {"output": result_text, "agent_id": "writer"}

    return {"error": "unknown agent", "output": None}

# Build the workflow
workflow = SequentialWorkflow(agent_executor=execute_crewai_step)
steps = [
    WorkflowStep(
        agent_id="researcher",
        action="Research the latest developments in quantum computing",
    ),
    WorkflowStep(
        agent_id="writer",
        action="Write a 500-word blog post based on the research findings",
    ),
]

result = asyncio.run(workflow.execute(steps))
print(f"Status: {result.status}")
for step_result in result.step_results:
    print(f"  {step_result.step.agent_id}: {step_result.output}")
```

---

## Pattern 2 — Parallel CrewAI Agent Execution

CrewAI runs tasks sequentially by default. agent-mesh-router's `ParallelWorkflow` executes
multiple CrewAI agents at the same time and collects all results.

```python
import asyncio
from agent_mesh_router import ParallelWorkflow, WorkflowStep

async def execute_crewai_step(step: WorkflowStep) -> dict[str, object]:
    # Map step.agent_id to your CrewAI agent and run it
    agent = get_crewai_agent(step.agent_id)
    task = Task(description=step.action, expected_output="Result", agent=agent)
    output = agent.execute_task(task)
    return {"output": output, "agent_id": step.agent_id}

# Run three research agents in parallel across different topics
workflow = ParallelWorkflow(agent_executor=execute_crewai_step, max_concurrency=3)
steps = [
    WorkflowStep(agent_id="researcher-a", action="Research quantum computing breakthroughs"),
    WorkflowStep(agent_id="researcher-b", action="Research AI regulation developments"),
    WorkflowStep(agent_id="researcher-c", action="Research climate tech investments"),
]
result = asyncio.run(workflow.execute(steps))
print(f"Completed {len(result.step_results)} parallel tasks in {result.elapsed_ms:.0f}ms")
```

---

## Pattern 3 — Competitive Execution (First-Agent-Wins)

Run the same task across multiple CrewAI agents and take the first successful response. Useful
for latency-sensitive tasks where you can afford to pay for a few concurrent calls.

```python
import asyncio
from agent_mesh_router import CompetitiveWorkflow, WorkflowStep

workflow = CompetitiveWorkflow(agent_executor=execute_crewai_step)
steps = [
    WorkflowStep(agent_id="fast-agent", action="Summarize this document: ..."),
    WorkflowStep(agent_id="quality-agent", action="Summarize this document: ..."),
    WorkflowStep(agent_id="backup-agent", action="Summarize this document: ..."),
]
# Returns as soon as the first agent completes successfully
result = asyncio.run(workflow.execute(steps))
print(f"Winner: {result.winning_agent_id}")
print(f"Result: {result.step_results[0].output}")
```

---

## Pattern 4 — Consensus Across CrewAI Agents

For high-stakes decisions, run the same prompt through multiple agents and require a quorum of
agreement before accepting the result.

```python
import asyncio
from agent_mesh_router import ConsensusWorkflow, WorkflowStep

workflow = ConsensusWorkflow(
    agent_executor=execute_crewai_step,
    quorum_fraction=0.67,   # require 2/3 agreement
)
steps = [
    WorkflowStep(agent_id="analyst-a", action="Should we approve this loan application?"),
    WorkflowStep(agent_id="analyst-b", action="Should we approve this loan application?"),
    WorkflowStep(agent_id="analyst-c", action="Should we approve this loan application?"),
]
result = asyncio.run(workflow.execute(steps))
print(f"Consensus reached: {result.consensus_reached}")
print(f"Decision: {result.consensus_output}")
```

---

## Pattern 5 — Backpressure and Queue Depth Monitoring

When CrewAI tasks are queued faster than agents can process them, backpressure prevents cascading
failures by throttling new task submissions.

```python
from agent_mesh_router import (
    AsyncMessageBroker,
    BackpressureMonitor,
    AdaptiveThrottle,
    PriorityMessageQueue,
    MessageEnvelope,
    MessageType,
    Priority,
    WARN_THRESHOLD,
    CRITICAL_THRESHOLD,
)
import uuid

# Set up the broker with backpressure monitoring
queue = PriorityMessageQueue(max_size=500)
broker = AsyncMessageBroker(queue=queue)
monitor = BackpressureMonitor(queue=queue, warn_threshold=WARN_THRESHOLD)
throttle = AdaptiveThrottle(monitor=monitor)

async def submit_crewai_task(description: str, priority: Priority = Priority.NORMAL) -> None:
    # Check backpressure before submitting
    await throttle.acquire()

    envelope = MessageEnvelope(
        message_id=str(uuid.uuid4()),
        message_type=MessageType.TASK,
        sender_id="orchestrator",
        payload={"description": description},
        priority=priority,
    )
    await broker.publish(envelope)

async def main() -> None:
    # High-priority task bypasses normal queue depth controls
    await submit_crewai_task("Urgent: analyze security incident", priority=Priority.HIGH)
    # Normal tasks are throttled when queue depth exceeds WARN_THRESHOLD
    await submit_crewai_task("Weekly report generation", priority=Priority.NORMAL)

asyncio.run(main())
```

---

## Pattern 6 — Protocol-Agnostic Transport for Distributed CrewAI

If your CrewAI agents run in separate processes or containers, agent-mesh-router provides
transport adapters so they can communicate over HTTP, gRPC, Redis, or Kafka without changing
agent code.

```python
from agent_mesh_router import HttpTransport, GrpcTransport
from agent_mesh_router import MessageEnvelope, MessageType, Priority
import uuid

# HTTP transport — send a task to a remote CrewAI agent endpoint
http_transport = HttpTransport(base_url="http://researcher-agent-service:8080")

envelope = MessageEnvelope(
    message_id=str(uuid.uuid4()),
    message_type=MessageType.TASK,
    sender_id="orchestrator",
    payload={"action": "Research quantum computing"},
    priority=Priority.NORMAL,
)

import asyncio
response = asyncio.run(http_transport.send(envelope))
print(f"Agent responded: {response.payload}")

# Switch to gRPC with one line — same agent code, different transport
grpc_transport = GrpcTransport(host="researcher-agent-service", port=50051)
response = asyncio.run(grpc_transport.send(envelope))
```

---

## What You Gain

1. **Circuit breakers on every CrewAI agent call** — if a downstream agent is slow or failing,
   the circuit opens and stops wasting time and money on calls that will fail.
2. **Parallel and competitive workflows** — CrewAI's sequential-by-default execution becomes
   parallel or competitive with a one-line change in the workflow type.
3. **Consensus patterns** — `ConsensusWorkflow` brings majority-vote decision making to any task
   where multiple agent perspectives should be reconciled before committing.
4. **Backpressure** — `BackpressureMonitor` and `AdaptiveThrottle` prevent task queues from
   growing without bound under load, protecting downstream agents from being overwhelmed.
5. **Protocol flexibility** — transport adapters (HTTP, gRPC, Redis, Kafka) let CrewAI agents
   communicate across process and service boundaries without changing agent code.
6. **Fleet registry and health monitoring** — `FleetRegistry` tracks which agents are healthy
   and can accept work, enabling `HealthMonitor` to take unhealthy agents out of rotation
   automatically.

## What You Keep

- All CrewAI agent definitions (`role`, `goal`, `backstory`, `tools`) are unchanged. agent-mesh-
  router sits outside CrewAI and wraps the execution layer, not the agent definition layer.
- CrewAI's task descriptions, expected outputs, and tool configurations remain in CrewAI.
- The `Process.sequential` and `Process.hierarchical` patterns you use in CrewAI today map
  directly to `SequentialWorkflow` and `HierarchicalWorkflow` in agent-mesh-router, so migration
  is mechanical rather than conceptual.
- Crews that do not need production routing — prototypes, local scripts — can continue using
  `crew.kickoff()` directly. agent-mesh-router is purely additive.
