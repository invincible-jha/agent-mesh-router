"""CompetitiveWorkflow — race N agents, take the first success.

The same logical task is dispatched to every agent in the ``steps`` list
simultaneously.  The first step to return a successful result wins; all
remaining in-flight tasks are cancelled.

This pattern is useful for:
- Reducing tail latency by hedging across multiple agents.
- Fallback: if the primary agent fails, a secondary can still win.
- A/B routing: dispatch to multiple model variants, keep the fastest.

Behaviour
---------
* All steps are launched concurrently via ``asyncio.gather`` with
  ``return_exceptions=True``.
* The *first successful* result is returned as the winner.
* If all steps fail, ``WorkflowStatus.FAILED`` is returned with every
  individual failure recorded in ``step_results``.
* Step-level timeouts (``step.timeout_seconds``) are honoured.

Example
-------
::

    import asyncio
    from agent_mesh_router.workflows.competitive import CompetitiveWorkflow
    from agent_mesh_router.workflows.base import WorkflowStep

    call_count = 0

    async def dispatch(step: WorkflowStep) -> dict[str, object]:
        global call_count
        call_count += 1
        if step.agent_id == "slow-agent":
            await asyncio.sleep(1.0)
        return {"agent": step.agent_id}

    wf = CompetitiveWorkflow(agent_executor=dispatch)
    steps = [
        WorkflowStep(agent_id="fast-agent", action="answer", step_id="s1"),
        WorkflowStep(agent_id="slow-agent", action="answer", step_id="s2"),
    ]
    result = asyncio.run(wf.execute(steps))
    print(result.status)  # WorkflowStatus.SUCCESS
    # Only the winning step appears in step_results
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

AgentCallable = Callable[[WorkflowStep], Awaitable[dict[str, object]]]


class CompetitiveWorkflow(WorkflowExecutor):
    """Dispatch the same task to N agents concurrently; take the first success.

    Parameters
    ----------
    agent_executor:
        Async callable ``(step) -> dict[str, object]`` invoked for each step.
    include_all_results:
        When True, ``WorkflowResult.step_results`` includes the outcomes of
        all steps (winners and losers).  When False (default), only the
        winning step is included (or all failures if all steps fail).
    """

    executor_name: str = "competitive"

    def __init__(
        self,
        agent_executor: AgentCallable,
        *,
        include_all_results: bool = False,
    ) -> None:
        self._agent_executor = agent_executor
        self._include_all_results = include_all_results

    async def execute(self, steps: list[WorkflowStep]) -> WorkflowResult:
        """Race all steps; return the first success.

        Parameters
        ----------
        steps:
            Steps representing competing agents/variants.  All steps
            receive identical dispatch (the step itself carries any
            per-agent parameterization via ``agent_id`` and ``params``).

        Returns
        -------
        WorkflowResult
            SUCCESS with the winning step result, or FAILED with all
            individual failures if every agent fails.
        """
        started_at = time.monotonic()

        if not steps:
            return WorkflowResult(
                step_results=[],
                status=WorkflowStatus.SUCCESS,
                duration_ms=0.0,
            )

        if len(steps) == 1:
            result = await self._run_step(steps[0])
            duration_ms = (time.monotonic() - started_at) * 1000.0
            return WorkflowResult(
                step_results=[result],
                status=WorkflowStatus.SUCCESS if result.success else WorkflowStatus.FAILED,
                duration_ms=duration_ms,
            )

        # Launch all competitors concurrently using asyncio tasks so we can
        # cancel losers when a winner is found.
        tasks: dict[asyncio.Task[StepResult], WorkflowStep] = {}
        for step in steps:
            task = asyncio.create_task(
                self._run_step(step), name=f"competitive-{step.step_id}"
            )
            tasks[task] = step

        winner_result: StepResult | None = None
        all_results: dict[str, StepResult] = {}
        pending: set[asyncio.Task[StepResult]] = set(tasks.keys())

        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for completed_task in done:
                try:
                    step_result: StepResult = completed_task.result()
                except Exception as exc:
                    # Task raised unexpectedly (not a step failure)
                    step = tasks[completed_task]
                    step_result = StepResult(
                        step_id=step.step_id,
                        success=False,
                        error=str(exc),
                    )

                all_results[step_result.step_id] = step_result

                if step_result.success and winner_result is None:
                    winner_result = step_result
                    logger.debug(
                        "Competitive winner: step %r.", step_result.step_id
                    )
                    # Cancel all remaining in-flight competitors
                    for remaining_task in pending:
                        remaining_task.cancel()
                    # Collect cancellation results
                    if pending:
                        cancel_outcomes = await asyncio.gather(
                            *pending, return_exceptions=True
                        )
                        for remaining_task, outcome in zip(
                            list(pending), cancel_outcomes
                        ):
                            cancelled_step = tasks[remaining_task]
                            if isinstance(outcome, StepResult):
                                all_results[cancelled_step.step_id] = outcome
                            else:
                                all_results[cancelled_step.step_id] = StepResult(
                                    step_id=cancelled_step.step_id,
                                    success=False,
                                    error="cancelled",
                                )
                    pending = set()
                    break

        duration_ms = (time.monotonic() - started_at) * 1000.0

        if winner_result is not None:
            if self._include_all_results:
                ordered = [all_results[s.step_id] for s in steps if s.step_id in all_results]
            else:
                ordered = [winner_result]
            return WorkflowResult(
                step_results=ordered,
                status=WorkflowStatus.SUCCESS,
                duration_ms=duration_ms,
            )

        # All failed
        ordered_failures = [all_results[s.step_id] for s in steps if s.step_id in all_results]
        return WorkflowResult(
            step_results=ordered_failures,
            status=WorkflowStatus.FAILED,
            duration_ms=duration_ms,
        )

    async def _run_step(self, step: WorkflowStep) -> StepResult:
        """Execute one competing step with optional timeout."""
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

        except asyncio.CancelledError:
            duration_ms = (time.monotonic() - step_started) * 1000.0
            return StepResult(
                step_id=step.step_id,
                success=False,
                duration_ms=duration_ms,
                error="cancelled",
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
            logger.warning("Competitive step %r failed: %s", step.step_id, exc)
            return StepResult(
                step_id=step.step_id,
                success=False,
                duration_ms=duration_ms,
                error=str(exc),
            )
