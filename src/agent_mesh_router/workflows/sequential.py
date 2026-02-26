"""SequentialWorkflow — execute workflow steps one after another.

Steps are executed in the order they appear in the ``steps`` list.
Execution stops immediately on the first failed step; remaining steps are
skipped and do not appear in ``WorkflowResult.step_results``.

When a step has a ``timeout_seconds`` value, ``asyncio.wait_for`` enforces
the per-step deadline.  Timeouts are treated as step failures (``success=False``).

Example
-------
::

    import asyncio
    from agent_mesh_router.workflows.sequential import SequentialWorkflow
    from agent_mesh_router.workflows.base import WorkflowStep

    async def run_agent_step(step: WorkflowStep) -> dict[str, object]:
        # Simulate agent call
        return {"result": f"completed {step.action}"}

    workflow = SequentialWorkflow(agent_executor=run_agent_step)

    steps = [
        WorkflowStep(agent_id="agent-a", action="fetch"),
        WorkflowStep(agent_id="agent-b", action="process"),
        WorkflowStep(agent_id="agent-c", action="store"),
    ]

    result = asyncio.run(workflow.execute(steps))
    print(result.status)      # WorkflowStatus.SUCCESS
    print(len(result.step_results))  # 3
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

from agent_mesh_router.workflows.base import (
    StepResult,
    WorkflowExecutor,
    WorkflowResult,
    WorkflowStatus,
    WorkflowStep,
)

logger = logging.getLogger(__name__)

# Signature for the pluggable agent-call function.
AgentCallable = Callable[[WorkflowStep], Awaitable[dict[str, object]]]


class SequentialWorkflow(WorkflowExecutor):
    """Execute workflow steps in order, stopping on first failure.

    Parameters
    ----------
    agent_executor:
        Async callable ``(step) -> dict[str, object]`` that dispatches the
        step to the target agent and returns its output.  Raising an
        exception is treated as a step failure.

    Example
    -------
    ::

        workflow = SequentialWorkflow(agent_executor=my_async_dispatch)
        result = await workflow.execute(steps)
    """

    executor_name: str = "sequential"

    def __init__(self, agent_executor: AgentCallable) -> None:
        self._agent_executor = agent_executor

    async def execute(self, steps: list[WorkflowStep]) -> WorkflowResult:
        """Execute ``steps`` sequentially, aborting on first failure.

        Parameters
        ----------
        steps:
            Steps to execute in order.  An empty list returns an immediate
            SUCCESS result with no step results.

        Returns
        -------
        WorkflowResult
            Aggregate result.  ``status`` is SUCCESS only when all steps
            succeed.  FAILED when any step fails (subsequent steps omitted).
        """
        started_at = time.monotonic()
        completed: list[StepResult] = []

        if not steps:
            return WorkflowResult(
                step_results=[],
                status=WorkflowStatus.SUCCESS,
                duration_ms=0.0,
            )

        for step in steps:
            step_result = await self._run_step(step)
            completed.append(step_result)

            if not step_result.success:
                logger.warning(
                    "SequentialWorkflow stopping after failed step %r: %s",
                    step.step_id,
                    step_result.error,
                )
                break

        duration_ms = (time.monotonic() - started_at) * 1000.0
        all_success = all(r.success for r in completed)
        status = WorkflowStatus.SUCCESS if all_success else WorkflowStatus.FAILED

        return WorkflowResult(
            step_results=completed,
            status=status,
            duration_ms=duration_ms,
        )

    async def _run_step(self, step: WorkflowStep) -> StepResult:
        """Execute a single step, honouring its timeout.

        Parameters
        ----------
        step:
            The step to execute.

        Returns
        -------
        StepResult
            Always returns a result; exceptions are caught and converted to
            failure results.
        """
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
            logger.debug(
                "Step %r succeeded in %.1f ms.", step.step_id, duration_ms
            )
            return StepResult(
                step_id=step.step_id,
                success=True,
                output=output,
                duration_ms=duration_ms,
            )

        except asyncio.TimeoutError:
            duration_ms = (time.monotonic() - step_started) * 1000.0
            error_msg = (
                f"Step timed out after {step.timeout_seconds}s."
            )
            logger.warning("Step %r timed out.", step.step_id)
            return StepResult(
                step_id=step.step_id,
                success=False,
                duration_ms=duration_ms,
                error=error_msg,
            )

        except Exception as exc:
            duration_ms = (time.monotonic() - step_started) * 1000.0
            logger.error(
                "Step %r raised an exception: %s", step.step_id, exc
            )
            return StepResult(
                step_id=step.step_id,
                success=False,
                duration_ms=duration_ms,
                error=str(exc),
            )
