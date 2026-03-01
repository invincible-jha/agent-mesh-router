#!/usr/bin/env python3
"""Example: Workflow Orchestration

Demonstrates Sequential, Parallel, and Competitive workflows for
multi-step agent task orchestration.

Usage:
    python examples/03_workflow_orchestration.py

Requirements:
    pip install agent-mesh-router
"""
from __future__ import annotations

import agent_mesh_router
from agent_mesh_router import (
    CompetitiveWorkflow,
    ParallelWorkflow,
    SequentialWorkflow,
    WorkflowStep,
    WorkflowStatus,
)


def make_step(name: str, output: str) -> WorkflowStep:
    """Create a step that returns a fixed output dict."""
    def handler(payload: dict[str, object]) -> dict[str, object]:
        return {"agent": name, "output": output}
    return WorkflowStep(name=name, handler=handler)


def print_result(label: str, result: object) -> None:
    print(f"\n{label}:")
    print(f"  Status: {result.status.value}")  # type: ignore[union-attr]
    for sr in result.step_results:  # type: ignore[union-attr]
        print(f"  [{sr.step_name}] {sr.output}")


def main() -> None:
    print(f"agent-mesh-router version: {agent_mesh_router.__version__}")

    payload: dict[str, object] = {"document": "quarterly-report.pdf"}

    # Sequential: steps run one after another
    sequential = SequentialWorkflow(steps=[
        make_step("fetch", "document retrieved"),
        make_step("parse", "text extracted: 2,400 words"),
        make_step("summarise", "summary: 3 key bullets"),
    ])
    seq_result = sequential.run(payload=payload)
    print_result("Sequential workflow", seq_result)
    assert seq_result.status == WorkflowStatus.SUCCESS

    # Parallel: all steps run concurrently
    parallel = ParallelWorkflow(steps=[
        make_step("sentiment", "positive: 0.78"),
        make_step("entities", "found: 12 entities"),
        make_step("topics", "topics: finance, growth"),
    ])
    par_result = parallel.run(payload=payload)
    print_result("Parallel workflow", par_result)
    assert par_result.status == WorkflowStatus.SUCCESS

    # Competitive: fastest / first successful result wins
    competitive = CompetitiveWorkflow(steps=[
        make_step("model-a", "summary variant A"),
        make_step("model-b", "summary variant B"),
    ])
    comp_result = competitive.run(payload=payload)
    print_result("Competitive workflow", comp_result)
    print(f"  Winner: {comp_result.winner_step_name}")


if __name__ == "__main__":
    main()
