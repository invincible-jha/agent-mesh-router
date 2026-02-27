"""YAMLWorkflow — declarative YAML-based workflow definition and execution.

Allows workflows to be declared as YAML documents and executed against a
context dict.  Supports three step patterns:

``task``
    A single synchronous task.  The handler is looked up from the context
    by the ``agent`` key.

``parallel``
    A list of sub-steps executed concurrently (all must succeed).

``conditional``
    An if/else branch.  The ``condition`` field is a dot-path expression
    evaluated against the context dict.

Commodity note
--------------
The YAML parser is PyYAML (standard library equivalent).  Conditional
evaluation is a simple dot-path key lookup — no expression engine.

Classes
-------
YAMLWorkflowError
    Base exception for this module.
YAMLParseError
    Raised when the YAML document cannot be parsed.
YAMLValidationError
    Raised when the YAML structure does not match the expected schema.
YAMLWorkflowStep
    A single declarative step from a YAML workflow definition.
YAMLWorkflowResult
    The outcome of executing a YAML workflow.
YAMLWorkflow
    Main workflow class.  Load from YAML, execute against a context dict.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class YAMLWorkflowError(Exception):
    """Base exception for all YAML workflow errors."""


class YAMLParseError(YAMLWorkflowError):
    """Raised when the YAML document cannot be parsed."""


class YAMLValidationError(YAMLWorkflowError):
    """Raised when the YAML document structure is invalid."""


# ---------------------------------------------------------------------------
# Step type constants
# ---------------------------------------------------------------------------

_STEP_TYPE_TASK = "task"
_STEP_TYPE_PARALLEL = "parallel"
_STEP_TYPE_CONDITIONAL = "conditional"
_VALID_STEP_TYPES = {_STEP_TYPE_TASK, _STEP_TYPE_PARALLEL, _STEP_TYPE_CONDITIONAL}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class YAMLWorkflowStep:
    """A single declarative step parsed from a YAML workflow definition.

    Attributes
    ----------
    name:
        Human-readable step identifier.
    step_type:
        One of ``"task"``, ``"parallel"``, or ``"conditional"``.
    agent:
        For ``task`` steps: the key in the context that holds the callable
        handler.  ``None`` for ``parallel`` and ``conditional`` steps.
    inputs:
        Key-value pairs resolved from the context at execution time.
    outputs:
        List of output keys to extract from the handler result dict
        and merge into the context.
    condition:
        Dot-path expression evaluated against the context for
        ``conditional`` steps.  ``None`` for other step types.
    if_steps:
        Sub-steps executed when *condition* is truthy.
    else_steps:
        Sub-steps executed when *condition* is falsy.
    sub_steps:
        Sub-steps for ``parallel`` steps.
    """

    name: str
    step_type: str
    agent: str | None = None
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: list[str] = field(default_factory=list)
    condition: str | None = None
    if_steps: list["YAMLWorkflowStep"] = field(default_factory=list)
    else_steps: list["YAMLWorkflowStep"] = field(default_factory=list)
    sub_steps: list["YAMLWorkflowStep"] = field(default_factory=list)


@dataclass
class YAMLWorkflowResult:
    """The outcome of executing a YAML workflow.

    Attributes
    ----------
    steps_completed:
        Names of steps that completed successfully.
    outputs:
        Merged output context produced by the workflow.
    duration_seconds:
        Wall-clock seconds for the full execution.
    errors:
        List of ``(step_name, error_message)`` pairs for failed steps.
    success:
        True when no steps failed.
    """

    steps_completed: list[str]
    outputs: dict[str, object]
    duration_seconds: float
    errors: list[tuple[str, str]] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Return True when the workflow completed with no errors."""
        return len(self.errors) == 0


# ---------------------------------------------------------------------------
# YAML parsing and schema helpers
# ---------------------------------------------------------------------------


def _load_yaml(yaml_str: str) -> object:
    """Parse a YAML string and return the Python object.

    Parameters
    ----------
    yaml_str:
        Raw YAML text.

    Returns
    -------
    object
        Python object (typically dict or list).

    Raises
    ------
    YAMLParseError
        If PyYAML cannot parse the input.
    """
    try:
        import yaml  # type: ignore[import]
    except ImportError as exc:
        raise YAMLWorkflowError(
            "PyYAML is required for YAML workflow support. "
            "Install it with: pip install pyyaml"
        ) from exc

    try:
        return yaml.safe_load(yaml_str)
    except Exception as exc:
        raise YAMLParseError(f"Failed to parse YAML: {exc}") from exc


def _require_str(mapping: dict[str, object], key: str, context: str) -> str:
    """Extract a required string field from a mapping.

    Parameters
    ----------
    mapping:
        Source dict.
    key:
        Required key name.
    context:
        Human-readable description used in error messages.

    Returns
    -------
    str
        The string value.

    Raises
    ------
    YAMLValidationError
        If the key is missing or not a string.
    """
    value = mapping.get(key)
    if value is None:
        raise YAMLValidationError(f"'{key}' is required in {context}.")
    if not isinstance(value, str):
        raise YAMLValidationError(
            f"'{key}' must be a string in {context}, got {type(value).__name__}."
        )
    return value


def _parse_step(raw_step: object, step_index: int) -> YAMLWorkflowStep:
    """Parse a single step dict from a YAML workflow into a YAMLWorkflowStep.

    Parameters
    ----------
    raw_step:
        Raw step dict from YAML parsing.
    step_index:
        Zero-based index (used in error messages).

    Returns
    -------
    YAMLWorkflowStep
        Parsed step.

    Raises
    ------
    YAMLValidationError
        If required fields are missing or types are wrong.
    """
    if not isinstance(raw_step, dict):
        raise YAMLValidationError(
            f"Step at index {step_index} must be a mapping, "
            f"got {type(raw_step).__name__}."
        )

    context_label = f"step[{step_index}]"
    name = _require_str(raw_step, "name", context_label)
    step_type = _require_str(raw_step, "type", f"step '{name}'")

    if step_type not in _VALID_STEP_TYPES:
        raise YAMLValidationError(
            f"Step '{name}' has unknown type '{step_type}'. "
            f"Valid types: {sorted(_VALID_STEP_TYPES)}."
        )

    # Common optional fields
    agent: str | None = raw_step.get("agent")  # type: ignore[assignment]
    raw_inputs = raw_step.get("inputs", {})
    raw_outputs = raw_step.get("outputs", [])

    if not isinstance(raw_inputs, dict):
        raise YAMLValidationError(
            f"Step '{name}' 'inputs' must be a mapping, got {type(raw_inputs).__name__}."
        )
    if not isinstance(raw_outputs, list):
        raise YAMLValidationError(
            f"Step '{name}' 'outputs' must be a list, got {type(raw_outputs).__name__}."
        )

    inputs: dict[str, str] = {str(k): str(v) for k, v in raw_inputs.items()}
    outputs: list[str] = [str(o) for o in raw_outputs]

    # Type-specific fields
    condition: str | None = None
    if_steps: list[YAMLWorkflowStep] = []
    else_steps: list[YAMLWorkflowStep] = []
    sub_steps: list[YAMLWorkflowStep] = []

    if step_type == _STEP_TYPE_CONDITIONAL:
        condition = _require_str(raw_step, "condition", f"step '{name}'")
        raw_if = raw_step.get("if_steps", [])
        raw_else = raw_step.get("else_steps", [])
        if_steps = [_parse_step(s, i) for i, s in enumerate(raw_if)]
        else_steps = [_parse_step(s, i) for i, s in enumerate(raw_else)]

    elif step_type == _STEP_TYPE_PARALLEL:
        raw_sub = raw_step.get("steps", [])
        if not isinstance(raw_sub, list):
            raise YAMLValidationError(
                f"Parallel step '{name}' requires a 'steps' list."
            )
        sub_steps = [_parse_step(s, i) for i, s in enumerate(raw_sub)]

    return YAMLWorkflowStep(
        name=name,
        step_type=step_type,
        agent=agent,
        inputs=inputs,
        outputs=outputs,
        condition=condition,
        if_steps=if_steps,
        else_steps=else_steps,
        sub_steps=sub_steps,
    )


def _parse_workflow_document(document: object) -> tuple[str, list[YAMLWorkflowStep]]:
    """Parse a full YAML workflow document.

    Expected structure::

        name: my-workflow
        steps:
          - name: step-one
            type: task
            agent: my_handler
            inputs:
              prompt: user_query
            outputs:
              - result

    Parameters
    ----------
    document:
        Python object produced by PyYAML.

    Returns
    -------
    tuple[str, list[YAMLWorkflowStep]]
        Workflow name and list of parsed steps.

    Raises
    ------
    YAMLValidationError
        If the document structure is invalid.
    """
    if not isinstance(document, dict):
        raise YAMLValidationError(
            f"Workflow document must be a YAML mapping, got {type(document).__name__}."
        )

    name = _require_str(document, "name", "workflow document")
    raw_steps = document.get("steps", [])

    if not isinstance(raw_steps, list):
        raise YAMLValidationError("'steps' must be a YAML list.")

    steps = [_parse_step(s, i) for i, s in enumerate(raw_steps)]
    return name, steps


# ---------------------------------------------------------------------------
# Context evaluation helpers
# ---------------------------------------------------------------------------


def _resolve_dot_path(context: dict[str, object], path: str) -> object:
    """Resolve a dot-separated key path against a context dict.

    Example: ``_resolve_dot_path({"user": {"name": "Alice"}}, "user.name")``
    returns ``"Alice"``.

    Parameters
    ----------
    context:
        Flat or nested context dict.
    path:
        Dot-separated key path.

    Returns
    -------
    object
        Resolved value, or ``None`` if the path does not exist.
    """
    current: object = context
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _is_truthy(value: object) -> bool:
    """Return True when *value* is truthy in a workflow condition context.

    None, False, empty string, 0, empty collections are all falsy.

    Parameters
    ----------
    value:
        Value to test.

    Returns
    -------
    bool
    """
    return bool(value)


def _resolve_inputs(
    step_inputs: dict[str, str],
    context: dict[str, object],
) -> dict[str, object]:
    """Resolve step inputs by looking up values from the execution context.

    Each value in *step_inputs* is treated as a context key.  If the key
    exists in *context* the resolved value is used; otherwise the key
    string itself is used as a literal value.

    Parameters
    ----------
    step_inputs:
        Mapping of parameter names to context key names.
    context:
        Current execution context.

    Returns
    -------
    dict[str, object]
        Resolved input values ready to pass to the handler.
    """
    resolved: dict[str, object] = {}
    for param_name, context_key in step_inputs.items():
        resolved[param_name] = context.get(context_key, context_key)
    return resolved


# ---------------------------------------------------------------------------
# Execution engine
# ---------------------------------------------------------------------------


class YAMLWorkflow:
    """A declarative YAML-based workflow engine.

    Load from a YAML string or file, then execute against a context dict.
    The context is mutated in-place as steps produce outputs.

    Parameters
    ----------
    name:
        Workflow name.
    steps:
        Parsed list of workflow steps.

    Example
    -------
    ::

        workflow = YAMLWorkflow.from_yaml(yaml_str)
        result = workflow.execute({"my_handler": my_fn, "user_input": "Hello"})
        print(result.outputs)
    """

    def __init__(self, name: str, steps: list[YAMLWorkflowStep]) -> None:
        self.name = name
        self.steps = steps

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "YAMLWorkflow":
        """Load a workflow from a YAML string.

        Parameters
        ----------
        yaml_str:
            YAML workflow definition.

        Returns
        -------
        YAMLWorkflow
            Parsed and validated workflow.

        Raises
        ------
        YAMLParseError
            If the YAML cannot be parsed.
        YAMLValidationError
            If the workflow structure is invalid.
        """
        document = _load_yaml(yaml_str)
        name, steps = _parse_workflow_document(document)
        return cls(name=name, steps=steps)

    @classmethod
    def from_file(cls, path: str | Path) -> "YAMLWorkflow":
        """Load a workflow from a YAML file.

        Parameters
        ----------
        path:
            Path to the ``.yaml`` or ``.yml`` file.

        Returns
        -------
        YAMLWorkflow
            Parsed and validated workflow.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        YAMLParseError
            If the file content cannot be parsed.
        YAMLValidationError
            If the workflow structure is invalid.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Workflow file not found: {path}")
        yaml_str = file_path.read_text(encoding="utf-8")
        return cls.from_yaml(yaml_str)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, context: dict[str, object]) -> YAMLWorkflowResult:
        """Execute the workflow against an execution context.

        The context is copied before execution so the original is not
        mutated.  Step handlers are callables stored in the context by
        the ``agent`` key name.

        Parameters
        ----------
        context:
            Execution context mapping.  Should contain:
            - Handler callables keyed by their agent name.
            - Input values keyed by their variable names.

        Returns
        -------
        YAMLWorkflowResult
            Execution outcome including completed steps, outputs, and errors.
        """
        start = time.monotonic()
        execution_context = dict(context)
        steps_completed: list[str] = []
        errors: list[tuple[str, str]] = []

        for step in self.steps:
            self._execute_step(
                step, execution_context, steps_completed, errors
            )

        duration = time.monotonic() - start
        outputs = {
            key: value
            for key, value in execution_context.items()
            if key not in context
        }
        # Also include any keys that changed value
        for key, value in execution_context.items():
            if key in context and context[key] != value:
                outputs[key] = value

        return YAMLWorkflowResult(
            steps_completed=steps_completed,
            outputs=outputs,
            duration_seconds=duration,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Internal execution helpers
    # ------------------------------------------------------------------

    def _execute_step(
        self,
        step: YAMLWorkflowStep,
        context: dict[str, object],
        steps_completed: list[str],
        errors: list[tuple[str, str]],
    ) -> None:
        """Dispatch a single step to the appropriate execution method."""
        if step.step_type == _STEP_TYPE_TASK:
            self._execute_task(step, context, steps_completed, errors)
        elif step.step_type == _STEP_TYPE_PARALLEL:
            self._execute_parallel(step, context, steps_completed, errors)
        elif step.step_type == _STEP_TYPE_CONDITIONAL:
            self._execute_conditional(step, context, steps_completed, errors)

    def _execute_task(
        self,
        step: YAMLWorkflowStep,
        context: dict[str, object],
        steps_completed: list[str],
        errors: list[tuple[str, str]],
    ) -> None:
        """Execute a single task step.

        Looks up the handler from the context by ``step.agent``, resolves
        inputs, calls the handler, then merges declared outputs back into
        the context.
        """
        if step.agent is None:
            errors.append((step.name, "Task step missing 'agent' key."))
            return

        handler = context.get(step.agent)
        if handler is None:
            errors.append(
                (step.name, f"Agent '{step.agent}' not found in execution context.")
            )
            return

        if not callable(handler):
            errors.append(
                (step.name, f"Agent '{step.agent}' in context is not callable.")
            )
            return

        resolved_inputs = _resolve_inputs(step.inputs, context)
        try:
            result = handler(**resolved_inputs)
        except Exception as exc:
            errors.append((step.name, f"Handler raised an exception: {exc}"))
            return

        # Merge outputs back into context
        if isinstance(result, dict):
            for output_key in step.outputs:
                if output_key in result:
                    context[output_key] = result[output_key]
        elif step.outputs:
            # Single non-dict result: assign to first declared output
            context[step.outputs[0]] = result

        steps_completed.append(step.name)

    def _execute_parallel(
        self,
        step: YAMLWorkflowStep,
        context: dict[str, object],
        steps_completed: list[str],
        errors: list[tuple[str, str]],
    ) -> None:
        """Execute all sub-steps of a parallel step sequentially.

        In this commodity implementation parallel sub-steps run in order
        (true async parallelism is an extension point for persistent
        backends or async runtimes).  All sub-steps run regardless of
        individual failures.
        """
        sub_completed: list[str] = []
        sub_errors: list[tuple[str, str]] = []

        for sub_step in step.sub_steps:
            self._execute_step(sub_step, context, sub_completed, sub_errors)

        steps_completed.extend(sub_completed)
        errors.extend(sub_errors)

        if not sub_errors:
            steps_completed.append(step.name)

    def _execute_conditional(
        self,
        step: YAMLWorkflowStep,
        context: dict[str, object],
        steps_completed: list[str],
        errors: list[tuple[str, str]],
    ) -> None:
        """Execute if_steps or else_steps based on the condition evaluation.

        The condition is a dot-path expression resolved against the current
        context.  The resolved value is evaluated for truthiness.
        """
        if step.condition is None:
            errors.append((step.name, "Conditional step missing 'condition' field."))
            return

        condition_value = _resolve_dot_path(context, step.condition)
        branch_steps = step.if_steps if _is_truthy(condition_value) else step.else_steps

        for branch_step in branch_steps:
            self._execute_step(branch_step, context, steps_completed, errors)

        steps_completed.append(step.name)
