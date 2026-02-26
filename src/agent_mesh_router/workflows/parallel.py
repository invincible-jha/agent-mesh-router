"""ParallelWorkflow — execute independent workflow steps concurrently.

All steps are dispatched via ``asyncio.gather`` so they run concurrently
within the same event loop.  Steps that specify ``depends_on`` are held
back until their dependencies complete; truly independent steps all start
at the same time.

Dependency resolution
---------------------
A simple topological ordering is computed before dispatch:

1. Steps with no ``depends_on`` start immediately in the first wave.
2. After the first wave completes, any step whose dependencies are all
   satisfied is added to the next wave.
3. If a dependency failed, downstream steps are skipped with a
   ``success=False`` result and ``error="dependency_failed"``.
4. Circular dependencies are detected and raise ``ValueError`` at call time.

Example
-------
::

    import asyncio
    from agent_mesh_router.workflows.parallel import ParallelWorkflow
    from agent_mesh_router.workflows.base import WorkflowStep

    async def dispatch(step: WorkflowStep) -> dict[str, object]:
        await asyncio.sleep(0.01)  # simulate I/O
        return {"step": step.step_id}

    wf = ParallelWorkflow(agent_executor=dispatch)
    steps = [
        WorkflowStep(agent_id="a", action="fetch", step_id="s1"),
        WorkflowStep(agent_id="b", action="fetch", step_id="s2"),
        WorkflowStep(agent_id="c", action="merge", step_id="s3",
                     depends_on=["s1", "s2"]),
    ]
    result = asyncio.run(wf.execute(steps))
    print(result.status)  # WorkflowStatus.SUCCESS
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable

from agent_mesh_router.workflows.base import (
    StepResult,
    WorkflowExecutor,
    WorkflowResult,
    WorkflowStatus,
    WorkflowStep,
)

logger = logging.getLogger(__name__)

AgentCallable = Callable[[WorkflowStep], Awaitable[dict[str, object]]]


class ParallelWorkflow(WorkflowExecutor):
    """Execute independent steps concurrently using asyncio.gather.

    Steps with ``depends_on`` lists wait for their dependencies to finish
    before launching.  A failing step marks all dependents as failed without
    running them.

    Parameters
    ----------
    agent_executor:
        Async callable ``(step) -> dict[str, object]`` that dispatches the
        step to the target agent and returns its output.
    """

    executor_name: str = "parallel"

    def __init__(self, agent_executor: AgentCallable) -> None:
        self._agent_executor = agent_executor

    async def execute(self, steps: list[WorkflowStep]) -> WorkflowResult:
        """Execute steps concurrently, respecting ``depends_on`` ordering.

        Parameters
        ----------
        steps:
            Steps to execute.  Independent steps run in parallel waves.

        Returns
        -------
        WorkflowResult
            Aggregate result.  Status is SUCCESS only when all steps succeed.
            PARTIAL when some succeed; FAILED when none succeed.

        Raises
        ------
        ValueError
            If a circular dependency is detected in ``depends_on`` references.
        """
        started_at = time.monotonic()

        if not steps:
            return WorkflowResult(
                step_results=[],
                status=WorkflowStatus.SUCCESS,
                duration_ms=0.0,
            )

        step_map: dict[str, WorkflowStep] = {s.step_id: s for s in steps}
        self._validate_dependencies(steps, step_map)

        # Track results keyed by step_id
        results: dict[str, StepResult] = {}

        # Track which step_ids have been launched (to avoid double-launch)
        launched: set[str] = set()
        remaining: list[WorkflowStep] = list(steps)

        while remaining:
            # Find steps whose dependencies are all satisfied (succeeded)
            ready: list[WorkflowStep] = []
            skip_now: list[WorkflowStep] = []

            for step in remaining:
                dep_results = [results.get(dep_id) for dep_id in step.depends_on]

                # All dependencies must have a result
                if not all(dep_results):
                    continue  # Not all deps completed yet

                # Check if any dependency failed
                failed_deps = [r for r in dep_results if r and not r.success]
                if failed_deps:
                    skip_now.append(step)
                else:
                    ready.append(step)

            if not ready and not skip_now:
                # No progress possible — shouldn't happen after validation
                logger.error(
                    "ParallelWorkflow stalled: no steps ready and no steps skipped."
                )
                break

            # Immediately mark dependency-failed steps as failed
            for step in skip_now:
                results[step.step_id] = StepResult(
                    step_id=step.step_id,
                    success=False,
                    duration_ms=0.0,
                    error="dependency_failed",
                )
                remaining.remove(step)
                logger.debug(
                    "Step %r skipped — dependency failed.", step.step_id
                )

            if ready:
                # Execute all ready steps concurrently
                step_results = await asyncio.gather(
                    *[self._run_step(s) for s in ready],
                    return_exceptions=False,
                )
                for step, result in zip(ready, step_results):
                    results[step.step_id] = result
                    remaining.remove(step)
                    launched.add(step.step_id)

        # Preserve original ordering of results
        ordered_results = [results[s.step_id] for s in steps if s.step_id in results]

        duration_ms = (time.monotonic() - started_at) * 1000.0
        all_success = all(r.success for r in ordered_results)
        any_success = any(r.success for r in ordered_results)

        if all_success:
            status = WorkflowStatus.SUCCESS
        elif any_success:
            status = WorkflowStatus.PARTIAL
        else:
            status = WorkflowStatus.FAILED

        return WorkflowResult(
            step_results=ordered_results,
            status=status,
            duration_ms=duration_ms,
        )

    async def _run_step(self, step: WorkflowStep) -> StepResult:
        """Execute a single step, honouring its timeout."""
        step_started = time.monotonic()
        try:
            if step.timeout_seconds is not None:
                output = await asyncio.wait_for(
                    self._agent_executor(step),
                    timeout=step.timeout_seconds,
                )
            else:
                output = await self._agent_executor(step)

            duration_ms = (time.monotonic() - step_started) * 1000.0
            return StepResult(
                step_id=step.step_id,
                success=True,
                output=output,
                duration_ms=duration_ms,
            )

        except asyncio.TimeoutError:
            duration_ms = (time.monotonic() - step_started) * 1000.0
            return StepResult(
                step_id=step.step_id,
                success=False,
                duration_ms=duration_ms,
                error=f"Step timed out after {step.timeout_seconds}s.",
            )

        except Exception as exc:
            duration_ms = (time.monotonic() - step_started) * 1000.0
            logger.error("Step %r failed: %s", step.step_id, exc)
            return StepResult(
                step_id=step.step_id,
                success=False,
                duration_ms=duration_ms,
                error=str(exc),
            )

    @staticmethod
    def _validate_dependencies(
        steps: list[WorkflowStep],
        step_map: dict[str, WorkflowStep],
    ) -> None:
        """Detect missing references and circular dependencies.

        Raises
        ------
        ValueError
            If any ``depends_on`` entry references an unknown step_id, or
            if a cycle is detected in the dependency graph.
        """
        for step in steps:
            for dep_id in step.depends_on:
                if dep_id not in step_map:
                    raise ValueError(
                        f"Step {step.step_id!r} depends on unknown step {dep_id!r}."
                    )

        # Kahn's algorithm for cycle detection
        in_degree: dict[str, int] = defaultdict(int)
        for step in steps:
            if step.step_id not in in_degree:
                in_degree[step.step_id] = 0
            for dep_id in step.depends_on:
                in_degree[step.step_id] += 1

        queue: list[str] = [
            step.step_id for step in steps if in_degree[step.step_id] == 0
        ]
        visited_count = 0

        # Build reverse edges: dep → [dependents]
        dependents: dict[str, list[str]] = defaultdict(list)
        for step in steps:
            for dep_id in step.depends_on:
                dependents[dep_id].append(step.step_id)

        while queue:
            node = queue.pop(0)
            visited_count += 1
            for dependent_id in dependents[node]:
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    queue.append(dependent_id)

        if visited_count != len(steps):
            raise ValueError(
                "Circular dependency detected in workflow steps."
            )
