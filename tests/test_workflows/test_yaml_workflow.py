"""Tests for YAMLWorkflow — declarative YAML-based workflow engine.

Covers:
- YAMLWorkflow.from_yaml() — valid YAML, parse error, validation error
- YAMLWorkflow.from_file() — file not found, valid file
- execute() — sequential tasks, parallel steps, conditional if/else
- Task steps — handler lookup, input resolution, output extraction
- Parallel steps — all succeed, partial failure
- Conditional steps — truthy branch, falsy branch, dot-path condition
- Error cases — missing agent, non-callable agent, handler exception
- YAMLWorkflowResult — success property, steps_completed, errors, outputs
- Edge cases — empty steps, no outputs declared, missing condition field
- _parse_step validation — missing name, missing type, invalid type
"""
from __future__ import annotations

import pathlib

import pytest

from agent_mesh_router.workflows.yaml_workflow import (
    YAMLParseError,
    YAMLValidationError,
    YAMLWorkflow,
    YAMLWorkflowError,
    YAMLWorkflowResult,
    YAMLWorkflowStep,
    _resolve_dot_path,
)


# ---------------------------------------------------------------------------
# Minimal YAML fixtures
# ---------------------------------------------------------------------------

_SIMPLE_SEQUENTIAL_YAML = """
name: simple-workflow
steps:
  - name: step-one
    type: task
    agent: handler_a
    inputs:
      prompt: user_input
    outputs:
      - result
"""

_PARALLEL_YAML = """
name: parallel-workflow
steps:
  - name: fan-out
    type: parallel
    steps:
      - name: sub-a
        type: task
        agent: handler_a
        outputs:
          - out_a
      - name: sub-b
        type: task
        agent: handler_b
        outputs:
          - out_b
"""

_CONDITIONAL_YAML = """
name: conditional-workflow
steps:
  - name: branch
    type: conditional
    condition: use_fast_path
    if_steps:
      - name: fast
        type: task
        agent: fast_handler
        outputs:
          - fast_result
    else_steps:
      - name: slow
        type: task
        agent: slow_handler
        outputs:
          - slow_result
"""

_EMPTY_STEPS_YAML = """
name: empty-workflow
steps: []
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler(return_value: object) -> object:
    """Return a simple callable that returns *return_value*."""
    def handler(**kwargs: object) -> object:
        return return_value
    return handler


def _make_dict_handler(output_key: str, output_value: object) -> object:
    """Return a handler that returns {output_key: output_value}."""
    def handler(**kwargs: object) -> dict[str, object]:
        return {output_key: output_value}
    return handler


# ---------------------------------------------------------------------------
# _resolve_dot_path
# ---------------------------------------------------------------------------


class TestResolveDotPath:
    def test_simple_key(self) -> None:
        assert _resolve_dot_path({"x": 42}, "x") == 42

    def test_nested_path(self) -> None:
        ctx = {"user": {"name": "Alice"}}
        assert _resolve_dot_path(ctx, "user.name") == "Alice"

    def test_missing_key_returns_none(self) -> None:
        assert _resolve_dot_path({"x": 1}, "y") is None

    def test_deep_path_missing_intermediate(self) -> None:
        assert _resolve_dot_path({"x": 1}, "x.y.z") is None

    def test_empty_context(self) -> None:
        assert _resolve_dot_path({}, "anything") is None


# ---------------------------------------------------------------------------
# YAMLWorkflow.from_yaml()
# ---------------------------------------------------------------------------


class TestFromYaml:
    def test_valid_yaml_returns_workflow(self) -> None:
        wf = YAMLWorkflow.from_yaml(_SIMPLE_SEQUENTIAL_YAML)
        assert isinstance(wf, YAMLWorkflow)
        assert wf.name == "simple-workflow"

    def test_steps_parsed(self) -> None:
        wf = YAMLWorkflow.from_yaml(_SIMPLE_SEQUENTIAL_YAML)
        assert len(wf.steps) == 1
        step = wf.steps[0]
        assert step.name == "step-one"
        assert step.step_type == "task"
        assert step.agent == "handler_a"

    def test_invalid_yaml_raises_parse_error(self) -> None:
        with pytest.raises(YAMLParseError):
            YAMLWorkflow.from_yaml("{{invalid: yaml: :")

    def test_missing_name_raises_validation_error(self) -> None:
        yaml_str = "steps:\n  - name: s\n    type: task\n"
        with pytest.raises(YAMLValidationError) as exc_info:
            YAMLWorkflow.from_yaml(yaml_str)
        assert "name" in str(exc_info.value)

    def test_missing_steps_gives_empty_list(self) -> None:
        wf = YAMLWorkflow.from_yaml("name: test")
        assert wf.steps == []

    def test_steps_not_list_raises(self) -> None:
        yaml_str = "name: test\nsteps: not_a_list\n"
        with pytest.raises(YAMLValidationError):
            YAMLWorkflow.from_yaml(yaml_str)

    def test_step_missing_name_raises(self) -> None:
        yaml_str = "name: test\nsteps:\n  - type: task\n"
        with pytest.raises(YAMLValidationError):
            YAMLWorkflow.from_yaml(yaml_str)

    def test_step_missing_type_raises(self) -> None:
        yaml_str = "name: test\nsteps:\n  - name: s\n"
        with pytest.raises(YAMLValidationError):
            YAMLWorkflow.from_yaml(yaml_str)

    def test_step_invalid_type_raises(self) -> None:
        yaml_str = "name: test\nsteps:\n  - name: s\n    type: unknown\n"
        with pytest.raises(YAMLValidationError) as exc_info:
            YAMLWorkflow.from_yaml(yaml_str)
        assert "unknown" in str(exc_info.value)

    def test_parallel_with_sub_steps_parsed(self) -> None:
        wf = YAMLWorkflow.from_yaml(_PARALLEL_YAML)
        assert wf.steps[0].step_type == "parallel"
        assert len(wf.steps[0].sub_steps) == 2

    def test_conditional_parsed_correctly(self) -> None:
        wf = YAMLWorkflow.from_yaml(_CONDITIONAL_YAML)
        step = wf.steps[0]
        assert step.step_type == "conditional"
        assert step.condition == "use_fast_path"
        assert len(step.if_steps) == 1
        assert len(step.else_steps) == 1

    def test_empty_steps_list(self) -> None:
        wf = YAMLWorkflow.from_yaml(_EMPTY_STEPS_YAML)
        assert wf.steps == []


# ---------------------------------------------------------------------------
# YAMLWorkflow.from_file()
# ---------------------------------------------------------------------------


class TestFromFile:
    def test_file_not_found_raises(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(FileNotFoundError):
            YAMLWorkflow.from_file(tmp_path / "nonexistent.yaml")

    def test_valid_file_loaded(self, tmp_path: pathlib.Path) -> None:
        yaml_file = tmp_path / "workflow.yaml"
        yaml_file.write_text(_SIMPLE_SEQUENTIAL_YAML, encoding="utf-8")
        wf = YAMLWorkflow.from_file(yaml_file)
        assert wf.name == "simple-workflow"

    def test_pathlib_path_accepted(self, tmp_path: pathlib.Path) -> None:
        yaml_file = tmp_path / "wf.yaml"
        yaml_file.write_text(_SIMPLE_SEQUENTIAL_YAML, encoding="utf-8")
        wf = YAMLWorkflow.from_file(yaml_file)
        assert isinstance(wf, YAMLWorkflow)

    def test_string_path_accepted(self, tmp_path: pathlib.Path) -> None:
        yaml_file = tmp_path / "wf.yaml"
        yaml_file.write_text(_SIMPLE_SEQUENTIAL_YAML, encoding="utf-8")
        wf = YAMLWorkflow.from_file(str(yaml_file))
        assert isinstance(wf, YAMLWorkflow)


# ---------------------------------------------------------------------------
# execute() — sequential task
# ---------------------------------------------------------------------------


class TestExecuteTask:
    def test_task_handler_called(self) -> None:
        called = []

        def handler(**kwargs: object) -> None:
            called.append(True)

        wf = YAMLWorkflow.from_yaml(_SIMPLE_SEQUENTIAL_YAML)
        wf.execute({"handler_a": handler, "user_input": "hello"})
        assert called

    def test_task_output_captured(self) -> None:
        def handler(**kwargs: object) -> dict[str, object]:
            return {"result": "the_result"}

        wf = YAMLWorkflow.from_yaml(_SIMPLE_SEQUENTIAL_YAML)
        result = wf.execute({"handler_a": handler, "user_input": "hello"})
        assert "result" in result.outputs
        assert result.outputs["result"] == "the_result"

    def test_step_completed_on_success(self) -> None:
        wf = YAMLWorkflow.from_yaml(_SIMPLE_SEQUENTIAL_YAML)
        result = wf.execute({
            "handler_a": _make_handler({"result": "ok"}),
            "user_input": "q",
        })
        assert "step-one" in result.steps_completed

    def test_missing_agent_in_context_is_error(self) -> None:
        wf = YAMLWorkflow.from_yaml(_SIMPLE_SEQUENTIAL_YAML)
        result = wf.execute({"user_input": "hello"})  # no handler_a
        assert not result.success
        assert any("handler_a" in err[1] for err in result.errors)

    def test_non_callable_agent_is_error(self) -> None:
        wf = YAMLWorkflow.from_yaml(_SIMPLE_SEQUENTIAL_YAML)
        result = wf.execute({"handler_a": "not_callable", "user_input": "hello"})
        assert not result.success

    def test_handler_exception_captured_in_errors(self) -> None:
        def broken_handler(**kwargs: object) -> None:
            raise RuntimeError("something went wrong")

        wf = YAMLWorkflow.from_yaml(_SIMPLE_SEQUENTIAL_YAML)
        result = wf.execute({"handler_a": broken_handler, "user_input": "hello"})
        assert not result.success
        assert any("something went wrong" in err[1] for err in result.errors)

    def test_input_resolved_from_context(self) -> None:
        received: dict[str, object] = {}

        def handler(**kwargs: object) -> None:
            received.update(kwargs)

        wf = YAMLWorkflow.from_yaml(_SIMPLE_SEQUENTIAL_YAML)
        wf.execute({"handler_a": handler, "user_input": "my prompt text"})
        # "prompt" maps to "user_input" in context
        assert received.get("prompt") == "my prompt text"

    def test_duration_positive(self) -> None:
        wf = YAMLWorkflow.from_yaml(_SIMPLE_SEQUENTIAL_YAML)
        result = wf.execute({"handler_a": _make_handler(None), "user_input": "x"})
        assert result.duration_seconds >= 0.0

    def test_empty_steps_workflow_succeeds(self) -> None:
        wf = YAMLWorkflow.from_yaml(_EMPTY_STEPS_YAML)
        result = wf.execute({})
        assert result.success
        assert result.steps_completed == []


# ---------------------------------------------------------------------------
# execute() — parallel
# ---------------------------------------------------------------------------


class TestExecuteParallel:
    def test_all_sub_steps_run(self) -> None:
        wf = YAMLWorkflow.from_yaml(_PARALLEL_YAML)
        result = wf.execute({
            "handler_a": _make_dict_handler("out_a", "value_a"),
            "handler_b": _make_dict_handler("out_b", "value_b"),
        })
        assert "sub-a" in result.steps_completed
        assert "sub-b" in result.steps_completed

    def test_parallel_step_name_in_completed_on_success(self) -> None:
        wf = YAMLWorkflow.from_yaml(_PARALLEL_YAML)
        result = wf.execute({
            "handler_a": _make_dict_handler("out_a", "a"),
            "handler_b": _make_dict_handler("out_b", "b"),
        })
        assert "fan-out" in result.steps_completed

    def test_outputs_from_sub_steps_captured(self) -> None:
        wf = YAMLWorkflow.from_yaml(_PARALLEL_YAML)
        result = wf.execute({
            "handler_a": _make_dict_handler("out_a", "aaa"),
            "handler_b": _make_dict_handler("out_b", "bbb"),
        })
        assert result.outputs.get("out_a") == "aaa"
        assert result.outputs.get("out_b") == "bbb"

    def test_sub_step_failure_reported(self) -> None:
        def broken(**kwargs: object) -> None:
            raise RuntimeError("broken")

        wf = YAMLWorkflow.from_yaml(_PARALLEL_YAML)
        result = wf.execute({
            "handler_a": broken,
            "handler_b": _make_dict_handler("out_b", "b"),
        })
        # At least one error, main step not in completed
        assert len(result.errors) > 0


# ---------------------------------------------------------------------------
# execute() — conditional
# ---------------------------------------------------------------------------


class TestExecuteConditional:
    def test_truthy_condition_runs_if_steps(self) -> None:
        wf = YAMLWorkflow.from_yaml(_CONDITIONAL_YAML)
        result = wf.execute({
            "use_fast_path": True,
            "fast_handler": _make_dict_handler("fast_result", "fast"),
            "slow_handler": _make_dict_handler("slow_result", "slow"),
        })
        assert "fast" in result.steps_completed
        assert "slow" not in result.steps_completed

    def test_falsy_condition_runs_else_steps(self) -> None:
        wf = YAMLWorkflow.from_yaml(_CONDITIONAL_YAML)
        result = wf.execute({
            "use_fast_path": False,
            "fast_handler": _make_dict_handler("fast_result", "fast"),
            "slow_handler": _make_dict_handler("slow_result", "slow"),
        })
        assert "slow" in result.steps_completed
        assert "fast" not in result.steps_completed

    def test_absent_condition_key_is_falsy(self) -> None:
        wf = YAMLWorkflow.from_yaml(_CONDITIONAL_YAML)
        result = wf.execute({
            "fast_handler": _make_dict_handler("fast_result", "fast"),
            "slow_handler": _make_dict_handler("slow_result", "slow"),
        })
        # use_fast_path not in context → falsy → else branch
        assert "slow" in result.steps_completed

    def test_conditional_step_name_in_completed(self) -> None:
        wf = YAMLWorkflow.from_yaml(_CONDITIONAL_YAML)
        result = wf.execute({
            "use_fast_path": True,
            "fast_handler": _make_dict_handler("fast_result", "f"),
            "slow_handler": _make_dict_handler("slow_result", "s"),
        })
        assert "branch" in result.steps_completed

    def test_dot_path_condition(self) -> None:
        yaml_str = """
name: dot-path-test
steps:
  - name: conditional-step
    type: conditional
    condition: flags.use_feature
    if_steps:
      - name: feature-on
        type: task
        agent: on_handler
        outputs: []
    else_steps:
      - name: feature-off
        type: task
        agent: off_handler
        outputs: []
"""
        wf = YAMLWorkflow.from_yaml(yaml_str)
        result = wf.execute({
            "flags": {"use_feature": True},
            "on_handler": _make_handler(None),
            "off_handler": _make_handler(None),
        })
        assert "feature-on" in result.steps_completed
        assert "feature-off" not in result.steps_completed


# ---------------------------------------------------------------------------
# YAMLWorkflowResult
# ---------------------------------------------------------------------------


class TestYAMLWorkflowResult:
    def test_success_true_when_no_errors(self) -> None:
        result = YAMLWorkflowResult(
            steps_completed=["a", "b"],
            outputs={},
            duration_seconds=0.1,
            errors=[],
        )
        assert result.success

    def test_success_false_when_errors_present(self) -> None:
        result = YAMLWorkflowResult(
            steps_completed=[],
            outputs={},
            duration_seconds=0.0,
            errors=[("step-one", "handler failed")],
        )
        assert not result.success

    def test_default_errors_empty(self) -> None:
        result = YAMLWorkflowResult(
            steps_completed=[],
            outputs={},
            duration_seconds=0.0,
        )
        assert result.errors == []


# ---------------------------------------------------------------------------
# YAMLWorkflowStep dataclass
# ---------------------------------------------------------------------------


class TestYAMLWorkflowStep:
    def test_construction(self) -> None:
        step = YAMLWorkflowStep(name="s", step_type="task", agent="my_agent")
        assert step.name == "s"
        assert step.step_type == "task"
        assert step.agent == "my_agent"

    def test_default_empty_collections(self) -> None:
        step = YAMLWorkflowStep(name="s", step_type="task")
        assert step.inputs == {}
        assert step.outputs == []
        assert step.if_steps == []
        assert step.else_steps == []
        assert step.sub_steps == []

    def test_condition_none_by_default(self) -> None:
        step = YAMLWorkflowStep(name="s", step_type="conditional")
        assert step.condition is None


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptions:
    def test_parse_error_is_workflow_error(self) -> None:
        exc = YAMLParseError("bad yaml")
        assert isinstance(exc, YAMLWorkflowError)

    def test_validation_error_is_workflow_error(self) -> None:
        exc = YAMLValidationError("bad structure")
        assert isinstance(exc, YAMLWorkflowError)
