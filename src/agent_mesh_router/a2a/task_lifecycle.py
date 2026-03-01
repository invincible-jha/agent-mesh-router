"""A2A Task Lifecycle — state machine and task manager.

Implements the A2A v0.3 task state machine with validated transitions.
The ``TaskManager`` is an in-memory store that enforces the state machine
on every transition, maintaining full history when
``state_transition_history`` is enabled.

Valid transitions per A2A spec
------------------------------
::

    submitted       -> working, canceled
    working         -> completed, failed, input-required, canceled
    input-required  -> working, canceled
    completed       -> (terminal — no transitions allowed)
    failed          -> (terminal — no transitions allowed)
    canceled        -> (terminal — no transitions allowed)
"""
from __future__ import annotations

from agent_mesh_router.a2a.models import (
    A2AMessage,
    A2ATask,
    Artifact,
    TaskState,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# State machine definition
# ---------------------------------------------------------------------------

_VALID_TRANSITIONS: dict[TaskState, frozenset[TaskState]] = {
    TaskState.SUBMITTED: frozenset({TaskState.WORKING, TaskState.CANCELED}),
    TaskState.WORKING: frozenset(
        {
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.INPUT_REQUIRED,
            TaskState.CANCELED,
        }
    ),
    TaskState.INPUT_REQUIRED: frozenset({TaskState.WORKING, TaskState.CANCELED}),
    TaskState.COMPLETED: frozenset(),  # terminal
    TaskState.FAILED: frozenset(),  # terminal
    TaskState.CANCELED: frozenset(),  # terminal
}

_TERMINAL_STATES: frozenset[TaskState] = frozenset(
    {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED}
)


class TaskNotFoundError(KeyError):
    """Raised when a task ID does not exist in the TaskManager."""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        super().__init__(f"Task {task_id!r} not found.")


class InvalidTaskTransitionError(ValueError):
    """Raised when a requested state transition is not permitted by the A2A spec."""

    def __init__(
        self,
        task_id: str,
        current_state: TaskState,
        requested_state: TaskState,
    ) -> None:
        self.task_id = task_id
        self.current_state = current_state
        self.requested_state = requested_state
        valid = sorted(s.value for s in _VALID_TRANSITIONS.get(current_state, frozenset()))
        super().__init__(
            f"Cannot transition task {task_id!r} from {current_state.value!r} "
            f"to {requested_state.value!r}. "
            f"Valid transitions from {current_state.value!r}: {valid}."
        )


class TaskManager:
    """In-memory A2A task store with enforced state machine transitions.

    Each ``TaskManager`` instance maintains its own task registry.  It is
    not thread-safe by default; wrap with a lock if used from multiple
    threads.

    Parameters
    ----------
    track_history:
        When True (default), every state transition is appended to the
        task's ``history`` list.  Disable to reduce memory usage when
        history is not required.

    Example
    -------
    ::

        manager = TaskManager()
        task = manager.create_task(A2AMessage(role="user", parts=[...]))
        task = manager.transition(task.id, TaskState.WORKING)
        task = manager.transition(task.id, TaskState.COMPLETED)
    """

    def __init__(self, *, track_history: bool = True) -> None:
        self._tasks: dict[str, A2ATask] = {}
        self._track_history = track_history

    def create_task(self, initial_message: A2AMessage) -> A2ATask:
        """Create a new task in the SUBMITTED state.

        Parameters
        ----------
        initial_message:
            The first user message that initiated this task.

        Returns
        -------
        A2ATask
            The newly created task with state SUBMITTED.
        """
        initial_status = TaskStatus(
            state=TaskState.SUBMITTED,
            message=initial_message,
        )
        task = A2ATask(status=initial_status)
        self._tasks[task.id] = task
        return task

    def transition(
        self,
        task_id: str,
        new_state: TaskState,
        message: A2AMessage | None = None,
    ) -> A2ATask:
        """Transition a task to a new state.

        Parameters
        ----------
        task_id:
            The ID of the task to transition.
        new_state:
            Target state.  Must be reachable from the current state per
            the A2A state machine.
        message:
            Optional message to associate with the new status (e.g. an
            agent response or error description).

        Returns
        -------
        A2ATask
            The updated task.

        Raises
        ------
        TaskNotFoundError
            If ``task_id`` is not registered.
        InvalidTaskTransitionError
            If the transition is not permitted by the A2A spec.
        """
        task = self._get_or_raise(task_id)
        current_state = task.status.state

        allowed = _VALID_TRANSITIONS.get(current_state, frozenset())
        if new_state not in allowed:
            raise InvalidTaskTransitionError(task_id, current_state, new_state)

        if self._track_history:
            task.history.append(task.status)

        task.status = TaskStatus(state=new_state, message=message)
        return task

    def add_artifact(self, task_id: str, artifact: Artifact) -> A2ATask:
        """Attach an artifact to a task.

        Parameters
        ----------
        task_id:
            ID of the task that produced the artifact.
        artifact:
            Artifact to attach.

        Returns
        -------
        A2ATask
            The updated task with the new artifact appended.

        Raises
        ------
        TaskNotFoundError
            If ``task_id`` is not registered.
        """
        task = self._get_or_raise(task_id)
        task.artifacts.append(artifact)
        return task

    def get_task(self, task_id: str) -> A2ATask:
        """Return the task for ``task_id``.

        Raises
        ------
        TaskNotFoundError
            If ``task_id`` is not registered.
        """
        return self._get_or_raise(task_id)

    def cancel_task(self, task_id: str) -> A2ATask:
        """Cancel a task if it is in a non-terminal state.

        Equivalent to calling ``transition(task_id, TaskState.CANCELED)``
        but provides a more explicit API and a friendlier error message when
        the task is already terminal.

        Parameters
        ----------
        task_id:
            ID of the task to cancel.

        Returns
        -------
        A2ATask
            The updated task in CANCELED state.

        Raises
        ------
        TaskNotFoundError
            If ``task_id`` is not registered.
        InvalidTaskTransitionError
            If the task is already in a terminal state.
        """
        return self.transition(task_id, TaskState.CANCELED)

    def list_tasks(self) -> list[A2ATask]:
        """Return all tasks currently held by this manager.

        Returns
        -------
        list[A2ATask]
            Snapshot list of all tasks (mutable references to live objects).
        """
        return list(self._tasks.values())

    def is_terminal(self, task_id: str) -> bool:
        """Return True if the task is in a terminal state.

        Raises
        ------
        TaskNotFoundError
            If ``task_id`` is not registered.
        """
        task = self._get_or_raise(task_id)
        return task.status.state in _TERMINAL_STATES

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_raise(self, task_id: str) -> A2ATask:
        """Return the task or raise TaskNotFoundError."""
        try:
            return self._tasks[task_id]
        except KeyError:
            raise TaskNotFoundError(task_id) from None
