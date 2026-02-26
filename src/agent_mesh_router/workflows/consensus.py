"""ConsensusWorkflow — collect quorum agreement across N agents.

The same task is dispatched to every agent in the ``steps`` list.
Execution waits until a configurable quorum of successful responses has
been received (default: 2/3 of all agents, rounded up).  Once quorum is
met, in-flight tasks are cancelled and the results are merged into a
single output dict.

Merge strategy
--------------
The default merge strategy (``merge_outputs``) performs a shallow union
of all successful ``StepResult.output`` dicts, with later results
overwriting earlier ones for duplicate keys.  Override ``merge_outputs``
in a subclass to implement custom strategies (majority vote, union, etc.).

Quorum calculation
------------------
``required_quorum = ceil(total_agents * quorum_fraction)``

With ``quorum_fraction=2/3`` (default) and 3 agents: quorum = 2.
With ``quorum_fraction=1.0`` and 3 agents: quorum = 3 (unanimous).
With ``quorum_fraction=0.5`` and 4 agents: quorum = 2.

Example
-------
::

    import asyncio
    from agent_mesh_router.workflows.consensus import ConsensusWorkflow
    from agent_mesh_router.workflows.base import WorkflowStep

    async def dispatch(step: WorkflowStep) -> dict[str, object]:
        return {"vote": "yes", "agent": step.agent_id}

    wf = ConsensusWorkflow(agent_executor=dispatch, quorum_fraction=2/3)
    steps = [
        WorkflowStep(agent_id="agent-1", action="vote", step_id="s1"),
        WorkflowStep(agent_id="agent-2", action="vote", step_id="s2"),
        WorkflowStep(agent_id="agent-3", action="vote", step_id="s3"),
    ]
    result = asyncio.run(wf.execute(steps))
    print(result.status)  # WorkflowStatus.SUCCESS
"""
from __future__ import annotations

import asyncio
import logging
import math
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


class ConsensusWorkflow(WorkflowExecutor):
    """Dispatch to N agents, wait for quorum, then merge results.

    Parameters
    ----------
    agent_executor:
        Async callable ``(step) -> dict[str, object]`` invoked for each step.
    quorum_fraction:
        Fraction of total agents that must succeed to declare consensus.
        Must be in the range (0.0, 1.0].  Defaults to ``2/3``.
    """

    executor_name: str = "consensus"

    def __init__(
        self,
        agent_executor: AgentCallable,
        *,
        quorum_fraction: float = 2 / 3,
    ) -> None:
        if not (0.0 < quorum_fraction <= 1.0):
            raise ValueError(
                f"quorum_fraction must be in (0.0, 1.0], got {quorum_fraction}."
            )
        self._agent_executor = agent_executor
        self._quorum_fraction = quorum_fraction

    async def execute(self, steps: list[WorkflowStep]) -> WorkflowResult:
        """Run all steps concurrently and wait for quorum.

        Parameters
        ----------
        steps:
            Steps representing agents that should reach consensus.

        Returns
        -------
        WorkflowResult
            SUCCESS with merged outputs once quorum is met.
            FAILED if insufficient agents succeed (< required quorum).
            PARTIAL if some (but not enough) agents succeed.
        """
        started_at = time.monotonic()

        if not steps:
            return WorkflowResult(
                step_results=[],
                status=WorkflowStatus.SUCCESS,
                duration_ms=0.0,
            )

        required_quorum = math.ceil(len(steps) * self._quorum_fraction)
        logger.debug(
            "ConsensusWorkflow: %d agents, quorum required = %d.",
            len(steps),
            required_quorum,
        )

        tasks: dict[asyncio.Task[StepResult], WorkflowStep] = {}
        for step in steps:
            task = asyncio.create_task(
                self._run_step(step), name=f"consensus-{step.step_id}"
            )
            tasks[task] = step

        successful_results: list[StepResult] = []
        all_results: dict[str, StepResult] = {}
        pending: set[asyncio.Task[StepResult]] = set(tasks.keys())
        quorum_met = False

        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for completed_task in done:
                try:
                    step_result: StepResult = completed_task.result()
                except Exception as exc:
                    step = tasks[completed_task]
                    step_result = StepResult(
                        step_id=step.step_id,
                        success=False,
                        error=str(exc),
                    )

                all_results[step_result.step_id] = step_result
                if step_result.success:
                    successful_results.append(step_result)

            # Check if quorum is already unachievable
            remaining_count = len(pending)
            current_successes = len(successful_results)
            max_possible = current_successes + remaining_count

            if current_successes >= required_quorum:
                quorum_met = True
                # Cancel remaining tasks — quorum reached
                for remaining_task in pending:
                    remaining_task.cancel()
                cancel_results = await asyncio.gather(*pending, return_exceptions=True)
                for remaining_task, outcome in zip(list(pending), cancel_results):
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

            if max_possible < required_quorum:
                # Impossible to reach quorum even if all remaining succeed
                logger.warning(
                    "ConsensusWorkflow: quorum unachievable "
                    "(%d successes, %d pending, %d required).",
                    current_successes,
                    remaining_count,
                    required_quorum,
                )
                # Cancel remaining tasks
                for remaining_task in pending:
                    remaining_task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                pending = set()
                break

        duration_ms = (time.monotonic() - started_at) * 1000.0
        ordered = [all_results[s.step_id] for s in steps if s.step_id in all_results]

        if quorum_met:
            merged_output = self.merge_outputs(successful_results)
            # Replace individual outputs with merged in a synthetic result
            consensus_result = StepResult(
                step_id="__consensus__",
                success=True,
                output=merged_output,
                duration_ms=duration_ms,
            )
            return WorkflowResult(
                step_results=ordered + [consensus_result],
                status=WorkflowStatus.SUCCESS,
                duration_ms=duration_ms,
            )

        any_success = bool(successful_results)
        status = WorkflowStatus.PARTIAL if any_success else WorkflowStatus.FAILED

        return WorkflowResult(
            step_results=ordered,
            status=status,
            duration_ms=duration_ms,
        )

    def merge_outputs(self, successful_results: list[StepResult]) -> dict[str, object]:
        """Merge outputs from all successful steps into a single dict.

        The default implementation does a shallow union; later results
        overwrite earlier ones for duplicate keys.  Subclass and override
        to implement majority-vote, averaging, or domain-specific merge.

        Parameters
        ----------
        successful_results:
            Step results where ``success`` is True.

        Returns
        -------
        dict[str, object]
            Merged output dict to attach to the consensus StepResult.
        """
        merged: dict[str, object] = {}
        for result in successful_results:
            if result.output:
                merged.update(result.output)
        merged["__quorum_size__"] = len(successful_results)
        return merged

    async def _run_step(self, step: WorkflowStep) -> StepResult:
        """Execute one consensus participant step."""
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
            logger.warning("Consensus step %r failed: %s", step.step_id, exc)
            return StepResult(
                step_id=step.step_id,
                success=False,
                duration_ms=duration_ms,
                error=str(exc),
            )
