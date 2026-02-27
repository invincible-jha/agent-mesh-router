"""Workflow coordination strategies for multi-agent task execution.

This package provides five concrete ``WorkflowExecutor`` implementations,
each designed for a different coordination pattern:

sequential
    Execute steps one after another; stop on first failure.
parallel
    Execute independent steps concurrently using asyncio.gather.
hierarchical
    Tree-structured execution: parents complete before children.
competitive
    Race N agents; take the first successful result.
consensus
    Dispatch to N agents; wait for a quorum (default 2/3) of successes.

All executors share the same dataclass types defined in ``base``:
``WorkflowStep``, ``StepResult``, ``WorkflowResult``, and ``WorkflowStatus``.

Example
-------
::

    from agent_mesh_router.workflows import (
        WorkflowStep,
        WorkflowResult,
        WorkflowStatus,
        SequentialWorkflow,
        ParallelWorkflow,
    )
"""
from __future__ import annotations

from agent_mesh_router.workflows.base import (
    StepResult,
    WorkflowExecutor,
    WorkflowResult,
    WorkflowStatus,
    WorkflowStep,
)
from agent_mesh_router.workflows.competitive import CompetitiveWorkflow
from agent_mesh_router.workflows.consensus import ConsensusWorkflow
from agent_mesh_router.workflows.hierarchical import HierarchicalWorkflow
from agent_mesh_router.workflows.parallel import ParallelWorkflow
from agent_mesh_router.workflows.sequential import SequentialWorkflow
from agent_mesh_router.workflows.yaml_workflow import (
    YAMLParseError,
    YAMLValidationError,
    YAMLWorkflow,
    YAMLWorkflowError,
    YAMLWorkflowResult,
    YAMLWorkflowStep,
)

__all__ = [
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
    # YAML workflow
    "YAMLParseError",
    "YAMLValidationError",
    "YAMLWorkflow",
    "YAMLWorkflowError",
    "YAMLWorkflowResult",
    "YAMLWorkflowStep",
]
