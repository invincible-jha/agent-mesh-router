import sys, asyncio
sys.path.insert(0, 'src')

from agent_mesh_router.workflows.base import WorkflowStep, WorkflowStatus
from agent_mesh_router.workflows.sequential import SequentialWorkflow
from agent_mesh_router.workflows.parallel import ParallelWorkflow
from agent_mesh_router.workflows.hierarchical import HierarchicalWorkflow
from agent_mesh_router.workflows.competitive import CompetitiveWorkflow
from agent_mesh_router.workflows.consensus import ConsensusWorkflow


async def test_sequential():
    call_order = []

    async def agent_exec(step: WorkflowStep) -> dict:
        call_order.append(step.step_id)
        return {'result': step.step_id}

    wf = SequentialWorkflow(agent_executor=agent_exec)
    steps = [
        WorkflowStep(agent_id='a', action='run', step_id='s1'),
        WorkflowStep(agent_id='b', action='run', step_id='s2'),
        WorkflowStep(agent_id='c', action='run', step_id='s3'),
    ]
    result = await wf.execute(steps)
    assert result.status == WorkflowStatus.SUCCESS
    assert call_order == ['s1', 's2', 's3'], f'Order: {call_order}'
    assert len(result.step_results) == 3
    print('SequentialWorkflow success: PASS')

    call_order.clear()

    async def fail_s2(step: WorkflowStep) -> dict:
        call_order.append(step.step_id)
        if step.step_id == 's2':
            raise RuntimeError('Step 2 failed')
        return {'ok': True}

    wf2 = SequentialWorkflow(agent_executor=fail_s2)
    result2 = await wf2.execute(steps)
    assert result2.status == WorkflowStatus.FAILED
    assert call_order == ['s1', 's2'], f'Should stop at s2, got {call_order}'
    assert len(result2.step_results) == 2
    print('SequentialWorkflow stop-on-failure: PASS')


async def test_parallel():
    started = []

    async def slow_exec(step: WorkflowStep) -> dict:
        started.append(step.step_id)
        await asyncio.sleep(0.01)
        return {'step': step.step_id}

    wf = ParallelWorkflow(agent_executor=slow_exec)
    steps = [
        WorkflowStep(agent_id='a', action='run', step_id='s1'),
        WorkflowStep(agent_id='b', action='run', step_id='s2'),
        WorkflowStep(agent_id='c', action='run', step_id='s3', depends_on=['s1', 's2']),
    ]
    result = await wf.execute(steps)
    assert result.status == WorkflowStatus.SUCCESS
    assert len(result.step_results) == 3
    print('ParallelWorkflow with deps: PASS')

    try:
        bad_steps = [
            WorkflowStep(agent_id='a', action='run', step_id='x', depends_on=['y']),
            WorkflowStep(agent_id='b', action='run', step_id='y', depends_on=['x']),
        ]
        await wf.execute(bad_steps)
        assert False, 'Should have raised ValueError'
    except ValueError:
        print('ParallelWorkflow circular dep detection: PASS')


async def test_hierarchical():
    exec_log = []

    async def exec_fn(step: WorkflowStep) -> dict:
        exec_log.append(step.step_id)
        return {'done': True}

    wf = HierarchicalWorkflow(agent_executor=exec_fn)
    steps = [
        WorkflowStep(agent_id='root', action='plan', step_id='root'),
        WorkflowStep(agent_id='child-a', action='work', step_id='child-a', depends_on=['root']),
        WorkflowStep(agent_id='child-b', action='work', step_id='child-b', depends_on=['root']),
        WorkflowStep(agent_id='leaf', action='merge', step_id='leaf', depends_on=['child-a', 'child-b']),
    ]
    result = await wf.execute(steps)
    assert result.status == WorkflowStatus.SUCCESS
    assert exec_log[0] == 'root', f'Root should execute first: {exec_log}'
    assert exec_log[-1] == 'leaf', f'Leaf should execute last: {exec_log}'
    print('HierarchicalWorkflow: PASS')

    async def fail_root(step: WorkflowStep) -> dict:
        if step.step_id == 'root':
            raise RuntimeError('Root failed')
        return {'ok': True}

    wf2 = HierarchicalWorkflow(agent_executor=fail_root)
    result2 = await wf2.execute(steps)
    assert result2.status == WorkflowStatus.FAILED
    child_results = {r.step_id: r for r in result2.step_results}
    assert child_results['child-a'].error == 'parent_step_failed'
    print('HierarchicalWorkflow parent cascade: PASS')


async def test_competitive():
    async def fast(step: WorkflowStep) -> dict:
        if step.agent_id == 'fast':
            await asyncio.sleep(0.001)
            return {'winner': True}
        await asyncio.sleep(5.0)
        return {'winner': False}

    wf = CompetitiveWorkflow(agent_executor=fast)
    steps = [
        WorkflowStep(agent_id='fast', action='run', step_id='s1'),
        WorkflowStep(agent_id='slow', action='run', step_id='s2'),
    ]
    result = await wf.execute(steps)
    assert result.status == WorkflowStatus.SUCCESS
    assert result.step_results[0].step_id == 's1'
    print('CompetitiveWorkflow: PASS')

    async def always_fail(step: WorkflowStep) -> dict:
        raise RuntimeError('Always fails')

    wf2 = CompetitiveWorkflow(agent_executor=always_fail)
    result2 = await wf2.execute(steps)
    assert result2.status == WorkflowStatus.FAILED
    print('CompetitiveWorkflow all-fail: PASS')


async def test_consensus():
    async def vote(step: WorkflowStep) -> dict:
        return {'vote': 'yes', 'agent': step.agent_id}

    wf = ConsensusWorkflow(agent_executor=vote, quorum_fraction=2/3)
    steps = [
        WorkflowStep(agent_id='a1', action='vote', step_id='s1'),
        WorkflowStep(agent_id='a2', action='vote', step_id='s2'),
        WorkflowStep(agent_id='a3', action='vote', step_id='s3'),
    ]
    result = await wf.execute(steps)
    assert result.status == WorkflowStatus.SUCCESS
    consensus_steps = [r for r in result.step_results if r.step_id == '__consensus__']
    assert len(consensus_steps) == 1
    assert consensus_steps[0].output.get('__quorum_size__') >= 2
    print('ConsensusWorkflow: PASS')

    fail_count = {'n': 0}

    async def mostly_fail(step: WorkflowStep) -> dict:
        fail_count['n'] += 1
        if fail_count['n'] <= 2:
            raise RuntimeError('fail')
        return {'ok': True}

    wf2 = ConsensusWorkflow(agent_executor=mostly_fail, quorum_fraction=2/3)
    result2 = await wf2.execute(steps)
    assert result2.status in (WorkflowStatus.FAILED, WorkflowStatus.PARTIAL)
    print('ConsensusWorkflow quorum-fail: PASS')


async def test_tracing():
    from agent_mesh_router.middleware.tracing import TracingMiddleware
    from agent_mesh_router.messages.envelope import MessageEnvelope

    received = []

    async def handler(env: MessageEnvelope) -> None:
        received.append(env)

    tracing = TracingMiddleware(handler, sample_rate=1.0)
    env = MessageEnvelope(sender='a', receiver='b', payload={'cmd': 'run'})
    await tracing(env)

    assert len(received) == 1
    assert 'span_id' in received[0].metadata
    assert received[0].metadata.get('sampled') == '1'
    assert 'trace_ingested_at' in received[0].metadata
    print('TracingMiddleware: PASS')


async def test_health_monitor():
    from agent_mesh_router.fleet.registry import FleetRegistry, AgentNode, AgentStatus
    from agent_mesh_router.fleet.health import HealthMonitor
    import time

    reg = FleetRegistry()
    node = AgentNode(agent_id='stale-agent', capabilities={'task'})
    node.last_heartbeat = time.time() - 100  # 100 seconds ago
    reg.register(node)

    monitor = HealthMonitor(reg, heartbeat_timeout_seconds=30.0)
    newly_unhealthy = monitor.check_all()
    assert 'stale-agent' in newly_unhealthy, f'Expected stale-agent: {newly_unhealthy}'
    assert reg.get('stale-agent').status == AgentStatus.UNHEALTHY

    # Restore via heartbeat
    monitor.record_heartbeat('stale-agent')
    assert reg.get('stale-agent').status == AgentStatus.HEALTHY
    print('HealthMonitor: PASS')


async def main():
    await test_sequential()
    await test_parallel()
    await test_hierarchical()
    await test_competitive()
    await test_consensus()
    await test_tracing()
    await test_health_monitor()
    print()
    print('All async tests: PASS')


asyncio.run(main())
