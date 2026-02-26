"""Tests for workflow executors: sequential, parallel, competitive, consensus."""
from __future__ import annotations

import asyncio

import pytest

from agent_mesh_router.workflows.base import (
    StepResult,
    WorkflowStatus,
    WorkflowStep,
)
from agent_mesh_router.workflows.competitive import CompetitiveWorkflow
from agent_mesh_router.workflows.consensus import ConsensusWorkflow
from agent_mesh_router.workflows.parallel import ParallelWorkflow
from agent_mesh_router.workflows.sequential import SequentialWorkflow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _step(
    step_id: str,
    agent_id: str = "agent-a",
    depends_on: list[str] | None = None,
    timeout_seconds: float | None = None,
) -> WorkflowStep:
    return WorkflowStep(
        step_id=step_id,
        agent_id=agent_id,
        action="test-action",
        depends_on=depends_on or [],
        timeout_seconds=timeout_seconds,
    )


async def _ok(step: WorkflowStep) -> dict[str, object]:
    return {"step_id": step.step_id, "ok": True}


async def _fail(step: WorkflowStep) -> dict[str, object]:
    raise ValueError(f"Step {step.step_id} deliberately failed")


async def _slow(step: WorkflowStep) -> dict[str, object]:
    await asyncio.sleep(10.0)
    return {"step_id": step.step_id}


# ---------------------------------------------------------------------------
# SequentialWorkflow
# ---------------------------------------------------------------------------


class TestSequentialWorkflow:
    @pytest.mark.asyncio
    async def test_empty_steps(self) -> None:
        wf = SequentialWorkflow(agent_executor=_ok)
        result = await wf.execute([])
        assert result.status == WorkflowStatus.SUCCESS
        assert result.step_results == []

    @pytest.mark.asyncio
    async def test_all_steps_succeed(self) -> None:
        wf = SequentialWorkflow(agent_executor=_ok)
        steps = [_step("s1"), _step("s2"), _step("s3")]
        result = await wf.execute(steps)
        assert result.status == WorkflowStatus.SUCCESS
        assert len(result.step_results) == 3
        assert all(r.success for r in result.step_results)

    @pytest.mark.asyncio
    async def test_stops_on_first_failure(self) -> None:
        call_order: list[str] = []

        async def mixed(step: WorkflowStep) -> dict[str, object]:
            call_order.append(step.step_id)
            if step.step_id == "s2":
                raise ValueError("boom")
            return {"ok": True}

        wf = SequentialWorkflow(agent_executor=mixed)
        steps = [_step("s1"), _step("s2"), _step("s3")]
        result = await wf.execute(steps)
        assert result.status == WorkflowStatus.FAILED
        assert "s3" not in call_order
        assert len(result.step_results) == 2

    @pytest.mark.asyncio
    async def test_timeout_treated_as_failure(self) -> None:
        wf = SequentialWorkflow(agent_executor=_slow)
        steps = [_step("s1", timeout_seconds=0.05)]
        result = await wf.execute(steps)
        assert result.status == WorkflowStatus.FAILED
        assert result.step_results[0].success is False
        assert "timed out" in (result.step_results[0].error or "").lower()

    @pytest.mark.asyncio
    async def test_step_result_has_duration(self) -> None:
        wf = SequentialWorkflow(agent_executor=_ok)
        result = await wf.execute([_step("s1")])
        assert result.step_results[0].duration_ms >= 0.0

    @pytest.mark.asyncio
    async def test_exception_captured_in_error_field(self) -> None:
        wf = SequentialWorkflow(agent_executor=_fail)
        result = await wf.execute([_step("s1")])
        assert result.step_results[0].success is False
        assert "deliberately failed" in (result.step_results[0].error or "")


# ---------------------------------------------------------------------------
# ParallelWorkflow
# ---------------------------------------------------------------------------


class TestParallelWorkflow:
    @pytest.mark.asyncio
    async def test_empty_steps(self) -> None:
        wf = ParallelWorkflow(agent_executor=_ok)
        result = await wf.execute([])
        assert result.status == WorkflowStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_all_steps_succeed(self) -> None:
        wf = ParallelWorkflow(agent_executor=_ok)
        steps = [_step("s1"), _step("s2"), _step("s3")]
        result = await wf.execute(steps)
        assert result.status == WorkflowStatus.SUCCESS
        assert len(result.step_results) == 3

    @pytest.mark.asyncio
    async def test_dependency_ordering(self) -> None:
        completed: list[str] = []

        async def track(step: WorkflowStep) -> dict[str, object]:
            completed.append(step.step_id)
            return {}

        wf = ParallelWorkflow(agent_executor=track)
        steps = [
            _step("s1"),
            _step("s2"),
            _step("s3", depends_on=["s1", "s2"]),
        ]
        result = await wf.execute(steps)
        assert result.status == WorkflowStatus.SUCCESS
        # s3 must be completed after s1 and s2.
        assert completed.index("s3") > completed.index("s1")
        assert completed.index("s3") > completed.index("s2")

    @pytest.mark.asyncio
    async def test_failed_dep_causes_downstream_skip(self) -> None:
        async def mixed(step: WorkflowStep) -> dict[str, object]:
            if step.step_id == "s1":
                raise RuntimeError("root failure")
            return {}

        wf = ParallelWorkflow(agent_executor=mixed)
        steps = [
            _step("s1"),
            _step("s2", depends_on=["s1"]),
        ]
        result = await wf.execute(steps)
        assert result.status == WorkflowStatus.FAILED
        s2_result = next(r for r in result.step_results if r.step_id == "s2")
        assert s2_result.error == "dependency_failed"

    @pytest.mark.asyncio
    async def test_partial_success_status(self) -> None:
        async def mixed(step: WorkflowStep) -> dict[str, object]:
            if step.step_id == "s1":
                raise RuntimeError("fail")
            return {}

        wf = ParallelWorkflow(agent_executor=mixed)
        steps = [_step("s1"), _step("s2")]
        result = await wf.execute(steps)
        assert result.status == WorkflowStatus.PARTIAL

    @pytest.mark.asyncio
    async def test_circular_dependency_raises(self) -> None:
        wf = ParallelWorkflow(agent_executor=_ok)
        steps = [
            _step("s1", depends_on=["s2"]),
            _step("s2", depends_on=["s1"]),
        ]
        with pytest.raises(ValueError, match="[Cc]ircular"):
            await wf.execute(steps)

    @pytest.mark.asyncio
    async def test_unknown_dependency_raises(self) -> None:
        wf = ParallelWorkflow(agent_executor=_ok)
        steps = [_step("s1", depends_on=["nonexistent"])]
        with pytest.raises(ValueError):
            await wf.execute(steps)

    @pytest.mark.asyncio
    async def test_step_timeout_in_parallel(self) -> None:
        async def mixed(step: WorkflowStep) -> dict[str, object]:
            if step.step_id == "slow":
                await asyncio.sleep(10.0)
            return {}

        wf = ParallelWorkflow(agent_executor=mixed)
        steps = [_step("slow", timeout_seconds=0.05), _step("fast")]
        result = await wf.execute(steps)
        slow = next(r for r in result.step_results if r.step_id == "slow")
        assert slow.success is False


# ---------------------------------------------------------------------------
# CompetitiveWorkflow
# ---------------------------------------------------------------------------


class TestCompetitiveWorkflow:
    @pytest.mark.asyncio
    async def test_empty_steps(self) -> None:
        wf = CompetitiveWorkflow(agent_executor=_ok)
        result = await wf.execute([])
        assert result.status == WorkflowStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_single_step_success(self) -> None:
        wf = CompetitiveWorkflow(agent_executor=_ok)
        result = await wf.execute([_step("s1")])
        assert result.status == WorkflowStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_single_step_failure(self) -> None:
        wf = CompetitiveWorkflow(agent_executor=_fail)
        result = await wf.execute([_step("s1")])
        assert result.status == WorkflowStatus.FAILED

    @pytest.mark.asyncio
    async def test_first_success_wins(self) -> None:
        async def race(step: WorkflowStep) -> dict[str, object]:
            if step.step_id == "fast":
                return {"winner": True}
            await asyncio.sleep(10.0)
            return {"winner": False}

        wf = CompetitiveWorkflow(agent_executor=race)
        steps = [_step("fast"), _step("slow")]
        result = await wf.execute(steps)
        assert result.status == WorkflowStatus.SUCCESS
        assert result.step_results[0].output == {"winner": True}

    @pytest.mark.asyncio
    async def test_all_failures_returns_failed(self) -> None:
        wf = CompetitiveWorkflow(agent_executor=_fail)
        steps = [_step("s1"), _step("s2")]
        result = await wf.execute(steps)
        assert result.status == WorkflowStatus.FAILED
        assert len(result.step_results) == 2

    @pytest.mark.asyncio
    async def test_include_all_results(self) -> None:
        async def mixed(step: WorkflowStep) -> dict[str, object]:
            if step.step_id == "s1":
                return {"ok": True}
            await asyncio.sleep(10.0)
            return {}

        wf = CompetitiveWorkflow(agent_executor=mixed, include_all_results=True)
        steps = [_step("s1"), _step("s2")]
        result = await wf.execute(steps)
        assert result.status == WorkflowStatus.SUCCESS
        # Include_all_results means both results present.
        assert len(result.step_results) >= 1

    @pytest.mark.asyncio
    async def test_timeout_marks_step_failed(self) -> None:
        wf = CompetitiveWorkflow(agent_executor=_slow)
        steps = [_step("s1", timeout_seconds=0.05)]
        result = await wf.execute(steps)
        assert result.status == WorkflowStatus.FAILED


# ---------------------------------------------------------------------------
# ConsensusWorkflow
# ---------------------------------------------------------------------------


class TestConsensusWorkflow:
    @pytest.mark.asyncio
    async def test_empty_steps(self) -> None:
        wf = ConsensusWorkflow(agent_executor=_ok)
        result = await wf.execute([])
        assert result.status == WorkflowStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_quorum_fraction_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            ConsensusWorkflow(agent_executor=_ok, quorum_fraction=0.0)

    @pytest.mark.asyncio
    async def test_quorum_fraction_above_one_raises(self) -> None:
        with pytest.raises(ValueError):
            ConsensusWorkflow(agent_executor=_ok, quorum_fraction=1.1)

    @pytest.mark.asyncio
    async def test_all_succeed_reaches_quorum(self) -> None:
        wf = ConsensusWorkflow(agent_executor=_ok, quorum_fraction=2 / 3)
        steps = [_step("s1"), _step("s2"), _step("s3")]
        result = await wf.execute(steps)
        assert result.status == WorkflowStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_consensus_result_appended(self) -> None:
        wf = ConsensusWorkflow(agent_executor=_ok, quorum_fraction=1.0)
        steps = [_step("s1"), _step("s2")]
        result = await wf.execute(steps)
        assert result.status == WorkflowStatus.SUCCESS
        consensus = next(
            (r for r in result.step_results if r.step_id == "__consensus__"), None
        )
        assert consensus is not None
        assert consensus.output is not None
        assert consensus.output.get("__quorum_size__") == 2

    @pytest.mark.asyncio
    async def test_insufficient_quorum_returns_partial(self) -> None:
        async def mixed(step: WorkflowStep) -> dict[str, object]:
            if step.step_id == "s1":
                return {}
            raise RuntimeError("fail")

        wf = ConsensusWorkflow(agent_executor=mixed, quorum_fraction=1.0)
        steps = [_step("s1"), _step("s2"), _step("s3")]
        result = await wf.execute(steps)
        # Only 1 succeeds, but quorum requires all 3 (fraction=1.0).
        assert result.status in {WorkflowStatus.PARTIAL, WorkflowStatus.FAILED}

    @pytest.mark.asyncio
    async def test_all_fail_returns_failed(self) -> None:
        wf = ConsensusWorkflow(agent_executor=_fail, quorum_fraction=0.5)
        steps = [_step("s1"), _step("s2")]
        result = await wf.execute(steps)
        assert result.status == WorkflowStatus.FAILED

    @pytest.mark.asyncio
    async def test_merge_outputs_shallow_union(self) -> None:
        async def produce(step: WorkflowStep) -> dict[str, object]:
            return {step.step_id: "done"}

        wf = ConsensusWorkflow(agent_executor=produce, quorum_fraction=1.0)
        steps = [_step("s1"), _step("s2")]
        result = await wf.execute(steps)
        consensus = next(r for r in result.step_results if r.step_id == "__consensus__")
        assert "s1" in (consensus.output or {})
        assert "s2" in (consensus.output or {})

    @pytest.mark.asyncio
    async def test_timeout_step_not_counted_in_quorum(self) -> None:
        async def mixed(step: WorkflowStep) -> dict[str, object]:
            if step.step_id == "slow":
                await asyncio.sleep(10.0)
            return {}

        wf = ConsensusWorkflow(agent_executor=mixed, quorum_fraction=1.0)
        steps = [_step("slow", timeout_seconds=0.05), _step("fast")]
        result = await wf.execute(steps)
        # quorum_fraction=1.0 requires all steps; slow times out → no quorum.
        assert result.status in {WorkflowStatus.PARTIAL, WorkflowStatus.FAILED}
