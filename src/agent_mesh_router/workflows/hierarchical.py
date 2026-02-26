"""HierarchicalWorkflow — tree-structured parent→child dispatch.

Execution model
---------------
Steps form an implicit tree via their ``depends_on`` relationships.  A
step is a *parent* if other steps declare it in their ``depends_on``.
A step is a *leaf* (or *child*) if nothing depends on it.

Execution proceeds level by level (breadth-first from roots):

1. Root steps (no ``depends_on``) execute first, in parallel within the wave.
2. When all roots finish, their direct dependents execute.
3. If a parent step fails, all its children are immediately marked failed
   without executing (cascading failure).

This produces a tree-structured execution graph where the parent node
dispatches work to children only after its own computation succeeds.

Example
-------
::

    import asyncio
    from agent_mesh_router.workflows.hierarchical import HierarchicalWorkflow
    from agent_mesh_router.workflows.base import WorkflowStep

    async def dispatch(step: WorkflowStep) -> dict[str, object]:
        return {"done": step.step_id}

    wf = HierarchicalWorkflow(agent_executor=dispatch)

    steps = [
        WorkflowStep(agent_id="orchestrator", action="plan",
                     step_id="root"),
        WorkflowStep(agent_id="worker-a", action="execute",
                     step_id="child-a", depends_on=["root"]),
        WorkflowStep(agent_id="worker-b", action="execute",
                     step_id="child-b", depends_on=["root"]),
        WorkflowStep(agent_id="aggregator", action="merge",
                     step_id="leaf", depends_on=["child-a", "child-b"]),
    ]
    result = asyncio.run(wf.execute(steps))
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


class HierarchicalWorkflow(WorkflowExecutor):
    """Tree-structured workflow: parent executes, then dispatches to children.

    Steps are grouped into levels by topological depth.  All steps at the
    same depth run concurrently; a step at depth N only runs once all its
    ``depends_on`` steps at depth N-1 have completed successfully.

    Parameters
    ----------
    agent_executor:
        Async callable ``(step) -> dict[str, object]`` used to execute each step.
    """

    executor_name: str = "hierarchical"

    def __init__(self, agent_executor: AgentCallable) -> None:
        self._agent_executor = agent_executor

    async def execute(self, steps: list[WorkflowStep]) -> WorkflowResult:
        """Execute steps in hierarchical wave order, respecting ``depends_on``.

        Parameters
        ----------
        steps:
            All steps in the workflow.  Tree structure is inferred from
            ``depends_on`` references.

        Returns
        -------
        WorkflowResult
            Aggregate result with all executed (and skipped) step outcomes.

        Raises
        ------
        ValueError
            If ``depends_on`` references an unknown step_id or a cycle exists.
        """
        started_at = time.monotonic()

        if not steps:
            return WorkflowResult(
                step_results=[],
                status=WorkflowStatus.SUCCESS,
                duration_ms=0.0,
            )

        step_map: dict[str, WorkflowStep] = {s.step_id: s for s in steps}
        self._validate(steps, step_map)

        levels = self._compute_levels(steps)
        results: dict[str, StepResult] = {}

        for level_steps in levels:
            # Separate ready steps from those with failed parents
            to_run: list[WorkflowStep] = []
            to_skip: list[WorkflowStep] = []

            for step in level_steps:
                failed_parents = [
                    dep_id
                    for dep_id in step.depends_on
                    if dep_id in results and not results[dep_id].success
                ]
                if failed_parents:
                    to_skip.append(step)
                else:
                    to_run.append(step)

            for step in to_skip:
                results[step.step_id] = StepResult(
                    step_id=step.step_id,
                    success=False,
                    duration_ms=0.0,
                    error="parent_step_failed",
                )
                logger.debug(
                    "Step %r skipped due to failed parent.", step.step_id
                )

            if to_run:
                wave_results = await asyncio.gather(
                    *[self._run_step(s) for s in to_run]
                )
                for step, result in zip(to_run, wave_results):
                    results[step.step_id] = result

        ordered = [results[s.step_id] for s in steps if s.step_id in results]
        duration_ms = (time.monotonic() - started_at) * 1000.0
        all_success = all(r.success for r in ordered)
        any_success = any(r.success for r in ordered)

        if all_success:
            status = WorkflowStatus.SUCCESS
        elif any_success:
            status = WorkflowStatus.PARTIAL
        else:
            status = WorkflowStatus.FAILED

        return WorkflowResult(
            step_results=ordered,
            status=status,
            duration_ms=duration_ms,
        )

    async def _run_step(self, step: WorkflowStep) -> StepResult:
        """Execute a single step with optional timeout."""
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
    def _compute_levels(steps: list[WorkflowStep]) -> list[list[WorkflowStep]]:
        """Group steps into depth-ordered levels for wave execution.

        Returns
        -------
        list[list[WorkflowStep]]
            Outer list is ordered by execution wave (level 0 runs first).
            Inner list contains all steps at that level.
        """
        step_map: dict[str, WorkflowStep] = {s.step_id: s for s in steps}
        depth: dict[str, int] = {}

        def get_depth(step_id: str, visited: set[str]) -> int:
            if step_id in depth:
                return depth[step_id]
            step = step_map[step_id]
            if not step.depends_on:
                depth[step_id] = 0
                return 0
            max_parent_depth = max(
                get_depth(dep_id, visited) for dep_id in step.depends_on
            )
            depth[step_id] = max_parent_depth + 1
            return depth[step_id]

        for step in steps:
            get_depth(step.step_id, set())

        max_depth = max(depth.values()) if depth else 0
        levels: list[list[WorkflowStep]] = [[] for _ in range(max_depth + 1)]
        for step in steps:
            levels[depth[step.step_id]].append(step)

        return levels

    @staticmethod
    def _validate(
        steps: list[WorkflowStep],
        step_map: dict[str, WorkflowStep],
    ) -> None:
        """Validate dependency references and detect cycles."""
        for step in steps:
            for dep_id in step.depends_on:
                if dep_id not in step_map:
                    raise ValueError(
                        f"Step {step.step_id!r} depends on unknown step {dep_id!r}."
                    )

        # DFS cycle detection
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {s.step_id: WHITE for s in steps}
        dependents: dict[str, list[str]] = defaultdict(list)
        for step in steps:
            for dep_id in step.depends_on:
                dependents[dep_id].append(step.step_id)

        def dfs(node: str) -> None:
            color[node] = GRAY
            for neighbor in dependents.get(node, []):
                if color[neighbor] == GRAY:
                    raise ValueError(
                        "Circular dependency detected in workflow steps."
                    )
                if color[neighbor] == WHITE:
                    dfs(neighbor)
            color[node] = BLACK

        for step in steps:
            if color[step.step_id] == WHITE:
                dfs(step.step_id)
