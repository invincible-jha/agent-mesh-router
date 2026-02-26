"""Verify the full public API is importable from the top-level package."""
import sys
sys.path.insert(0, 'src')

import agent_mesh_router as amr

# Verify version
assert amr.__version__ == '0.1.0'

# Verify messages
assert hasattr(amr, 'MessageEnvelope')
assert hasattr(amr, 'MessageType')
assert hasattr(amr, 'Priority')
assert hasattr(amr, 'HandoffMetrics')
assert hasattr(amr, 'MessageSerializer')

# Verify routing
assert hasattr(amr, 'RoutingTable')
assert hasattr(amr, 'AgentRecord')
assert hasattr(amr, 'RouteResolver')
assert hasattr(amr, 'RoundRobinRouter')
assert hasattr(amr, 'LeastLoadedRouter')
assert hasattr(amr, 'CostAwareRouter')
assert hasattr(amr, 'CapabilityMatchRouter')
assert hasattr(amr, 'CompositeRouter')

# Verify broker
assert hasattr(amr, 'AsyncMessageBroker')
assert hasattr(amr, 'PriorityMessageQueue')
assert hasattr(amr, 'DeadLetterQueue')
assert hasattr(amr, 'DeadLetterEntry')
assert hasattr(amr, 'QueueFullError')
assert hasattr(amr, 'TopicManager')
assert hasattr(amr, 'TopicSubscription')

# Verify fleet
assert hasattr(amr, 'AgentNode')
assert hasattr(amr, 'AgentStatus')
assert hasattr(amr, 'FleetRegistry')
assert hasattr(amr, 'HealthMonitor')
assert hasattr(amr, 'LoadBalancer')
assert hasattr(amr, 'Strategy')

# Verify workflows
assert hasattr(amr, 'WorkflowStep')
assert hasattr(amr, 'StepResult')
assert hasattr(amr, 'WorkflowResult')
assert hasattr(amr, 'WorkflowStatus')
assert hasattr(amr, 'WorkflowExecutor')
assert hasattr(amr, 'SequentialWorkflow')
assert hasattr(amr, 'ParallelWorkflow')
assert hasattr(amr, 'HierarchicalWorkflow')
assert hasattr(amr, 'CompetitiveWorkflow')
assert hasattr(amr, 'ConsensusWorkflow')

# Verify backpressure
assert hasattr(amr, 'BackpressureMonitor')
assert hasattr(amr, 'AdaptiveThrottle')
assert hasattr(amr, 'WARN_THRESHOLD')
assert hasattr(amr, 'CRITICAL_THRESHOLD')
assert hasattr(amr, 'QueueMetrics')
assert amr.WARN_THRESHOLD == 0.7
assert amr.CRITICAL_THRESHOLD == 0.9

# Verify adapters
assert hasattr(amr, 'HttpTransport')
assert hasattr(amr, 'HttpTransportError')
assert hasattr(amr, 'GrpcTransport')
assert hasattr(amr, 'GrpcTransportError')

# Verify middleware
assert hasattr(amr, 'TracingMiddleware')

# Verify plugins
assert hasattr(amr, 'PluginRegistry')
assert hasattr(amr, 'PluginNotFoundError')
assert hasattr(amr, 'PluginAlreadyRegisteredError')

# Quick instantiation checks
env = amr.MessageEnvelope(sender='a', receiver='b', payload={'x': 1})
assert env.sender == 'a'

table = amr.RoutingTable()
table.register_agent('test-agent', capabilities=['nlp'])
assert 'test-agent' in table

registry = amr.FleetRegistry()
node = amr.AgentNode(agent_id='node-1', capabilities={'nlp'})
registry.register(node)
available = registry.get_available('nlp')
assert len(available) == 1

step = amr.WorkflowStep(agent_id='a', action='run', step_id='s1')
assert step.action == 'run'

monitor = amr.BackpressureMonitor()
monitor.register_queue('main', max_depth=100)
monitor.update_depth('main', 50)
assert abs(monitor.pressure_score() - 0.5) < 0.001

tm = amr.TopicManager()
tm.subscribe('agent-x', 'events.*')
hits = tm.publish('events.start', {})
assert 'agent-x' in hits

print('Full public API verification: PASS')
print(f'agent_mesh_router v{amr.__version__} — all {len(amr.__all__)} exports verified')
