"""Workflow abstractions — steps, results, and the executor ABC.

This module defines the core dataclasses and the abstract base class that
all workflow executors must implement.  Concrete executors (sequential,
parallel, hierarchical, competitive, consensus) live in sibling modules.

Dataclasses
-----------
WorkflowStep
    A single unit of work: which agent to call, what action to invoke,
    what parameters to pass, and how long to allow.
StepResult
    The outcome of executing one WorkflowStep.
WorkflowResult
    The aggregate outcome of executing all steps in a workflow.

Abstract class
--------------
WorkflowExecutor
    Subclasses implement ``execute(steps) -> WorkflowResult`` with a
    specific coordination pattern (sequential, parallel, etc.).
"""
from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar


class WorkflowStatus(str, Enum):
    """Overall status of a workflow execution.

    Values
    ------
    SUCCESS:
        All required steps completed without error.
    PARTIAL:
        Some steps succeeded; others failed (relevant for parallel/consensus).
    FAILED:
        The workflow could not be completed successfully.
    CANCELLED:
        Execution was cancelled before all steps could run.
    """

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowStep:
    """A single step within a workflow.

    Attributes
    ----------
    step_id:
        Unique identifier for this step within the workflow.
        Auto-generated as a UUID4 string if not provided.
    agent_id:
        The ID of the agent responsible for executing this step.
    action:
        The name of the action or operation the agent should perform
        (e.g. ``"summarize"``, ``"translate"``, ``"analyze"``).
    params:
        Keyword arguments passed to the agent when invoking the action.
    timeout_seconds:
        Maximum wall-clock seconds allowed for this step.
        ``None`` means no per-step timeout (workflow-level timeout applies).
    depends_on:
        List of ``step_id`` values that must complete successfully before
        this step may execute.  Empty list means no dependencies.
    """

    agent_id: str
    action: str
    params: dict[str, object] = field(default_factory=dict)
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timeout_seconds: float | None = None
    depends_on: list[str] = field(default_factory=list)


@dataclass
class StepResult:
    """The outcome of executing a single WorkflowStep.

    Attributes
    ----------
    step_id:
        Matches the ``WorkflowStep.step_id`` that produced this result.
    success:
        True when the step completed without error.
    output:
        The data returned by the agent for this step.
        ``None`` when the step failed before producing output.
    duration_ms:
        Wall-clock milliseconds taken to execute this step.
    error:
        Error description string when ``success`` is False; ``None`` otherwise.
    """

    step_id: str
    success: bool
    output: dict[str, object] | None = None
    duration_ms: float = 0.0
    error: str | None = None


@dataclass
class WorkflowResult:
    """Aggregate outcome of a complete workflow execution.

    Attributes
    ----------
    workflow_id:
        Unique identifier for this workflow run.
        Auto-generated as a UUID4 string if not provided.
    step_results:
        Ordered list of individual step outcomes.
    status:
        Overall workflow status.
    duration_ms:
        Total wall-clock milliseconds from first step start to last step end.
    """

    step_results: list[StepResult]
    status: WorkflowStatus
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    duration_ms: float = 0.0

    def successful_steps(self) -> list[StepResult]:
        """Return only the step results where ``success`` is True."""
        return [r for r in self.step_results if r.success]

    def failed_steps(self) -> list[StepResult]:
        """Return only the step results where ``success`` is False."""
        return [r for r in self.step_results if not r.success]

    def is_success(self) -> bool:
        """Return True when the overall workflow status is SUCCESS."""
        return self.status == WorkflowStatus.SUCCESS


class WorkflowExecutor(ABC):
    """Abstract base class for all workflow coordination strategies.

    Subclasses implement a specific execution pattern such as sequential,
    parallel fan-out, hierarchical tree dispatch, competitive racing, or
    consensus quorum collection.

    Each executor is stateless by default — a new workflow run is started
    each time ``execute`` is called.  Subclasses may introduce internal
    state (e.g., round-robin counters) as long as concurrent ``execute``
    calls remain safe.

    Subclass contract
    -----------------
    * ``execute`` must return a ``WorkflowResult`` whose ``workflow_id``
      is unique for each call.
    * ``execute`` must populate ``duration_ms`` on the returned result.
    * Steps that fail should be represented as ``StepResult(success=False)``
      rather than raising exceptions from ``execute`` itself.
    """

    #: Human-readable label for this executor type (set by subclasses).
    executor_name: ClassVar[str] = "base"

    @abstractmethod
    async def execute(self, steps: list[WorkflowStep]) -> WorkflowResult:
        """Execute the given workflow steps and return an aggregate result.

        Parameters
        ----------
        steps:
            Ordered list of workflow steps.  The interpretation of ordering
            depends on the concrete executor strategy.

        Returns
        -------
        WorkflowResult
            Aggregate outcome including per-step results, overall status,
            and total wall-clock duration.
        """

    # ------------------------------------------------------------------
    # Shared helpers available to all concrete executors
    # ------------------------------------------------------------------

    @staticmethod
    def _make_result(
        step_results: list[StepResult],
        started_at: float,
    ) -> WorkflowResult:
        """Build a WorkflowResult, computing status and duration automatically.

        Parameters
        ----------
        step_results:
            The list of completed step results.
        started_at:
            Unix epoch (seconds) when the workflow began executing.

        Returns
        -------
        WorkflowResult
            Fully populated result with computed status and duration.
        """
        duration_ms = (time.monotonic() - started_at) * 1000.0

        all_success = all(r.success for r in step_results)
        any_success = any(r.success for r in step_results)

        if all_success:
            status = WorkflowStatus.SUCCESS
        elif any_success:
            status = WorkflowStatus.PARTIAL
        else:
            status = WorkflowStatus.FAILED

        return WorkflowResult(
            step_results=step_results,
            status=status,
            duration_ms=duration_ms,
        )
