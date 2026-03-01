"""ACP-to-A2A Bridge — translates between AumOS MessageEnvelope and A2A tasks.

The ``ACPBridge`` converts between the AumOS Agent Communication Protocol
(ACP) ``MessageEnvelope`` dataclass and the A2A protocol's ``A2ATask`` /
``A2AMessage`` models.

Conversion rules
----------------
envelope -> task:
    - ``envelope.sender`` becomes task metadata ``acp_sender``
    - ``envelope.receiver`` becomes task metadata ``acp_receiver``
    - ``envelope.payload`` is wrapped in an ``A2AMessage`` with role ``"user"``
    - ``envelope.trace_id`` is stored in task metadata
    - Task is created in SUBMITTED state

task -> envelope:
    - The task's current status message (if any) becomes the envelope payload
    - Task ID is stored in envelope metadata
    - Task state is stored in envelope metadata

payload -> message part:
    - If payload has a ``"text"`` key, a text MessagePart is created
    - Otherwise the payload is JSON-encoded into the text field
"""
from __future__ import annotations

import json

from agent_mesh_router.a2a.models import (
    A2AMessage,
    A2ATask,
    Artifact,
    MessagePart,
    TaskState,
    TaskStatus,
)
from agent_mesh_router.messages.envelope import MessageEnvelope


class ACPBridgeError(ValueError):
    """Raised when a conversion between ACP and A2A formats fails."""


class ACPBridge:
    """Translates between ACP ``MessageEnvelope`` and A2A ``A2ATask`` messages.

    The bridge is stateless — a single instance can be shared across many
    conversions without any thread-safety concerns.

    Example
    -------
    ::

        bridge = ACPBridge()

        # Convert an ACP envelope to an A2A task.
        envelope = MessageEnvelope(sender="agent-a", receiver="agent-b", payload={...})
        task = bridge.envelope_to_task(envelope)

        # Convert the task back to an ACP envelope.
        reply_envelope = bridge.task_to_envelope(task, sender="agent-b", receiver="agent-a")
    """

    def envelope_to_task(self, envelope: MessageEnvelope) -> A2ATask:
        """Convert an ACP MessageEnvelope into an A2A task in SUBMITTED state.

        Parameters
        ----------
        envelope:
            The ACP envelope to convert.

        Returns
        -------
        A2ATask
            A new task whose initial message reflects the envelope's payload.
            Task metadata includes ACP routing information (sender, receiver,
            trace_id, message_id).
        """
        a2a_message = self.message_to_part(dict(envelope.payload))
        initial_status = TaskStatus(
            state=TaskState.SUBMITTED,
            message=a2a_message,
        )
        task_metadata: dict[str, object] = {
            "acp_sender": envelope.sender,
            "acp_receiver": envelope.receiver,
            "acp_trace_id": envelope.trace_id,
            "acp_message_id": envelope.message_id,
        }
        return A2ATask(
            status=initial_status,
            metadata=task_metadata,
        )

    def task_to_envelope(
        self,
        task: A2ATask,
        sender: str,
        receiver: str,
    ) -> MessageEnvelope:
        """Convert an A2A task into an ACP MessageEnvelope.

        The current status message (if present) is used as the envelope
        payload.  When no message is available, an empty payload carrying
        only the task state is produced.

        Parameters
        ----------
        task:
            The A2A task to convert.
        sender:
            ACP sender agent ID for the outgoing envelope.
        receiver:
            ACP receiver agent ID for the outgoing envelope.

        Returns
        -------
        MessageEnvelope
            ACP envelope representing the task's current state.
        """
        if not sender or not sender.strip():
            raise ACPBridgeError("sender must be a non-empty string.")
        if not receiver or not receiver.strip():
            raise ACPBridgeError("receiver must be a non-empty string.")

        payload: dict[str, object] = {
            "a2a_task_id": task.id,
            "a2a_state": task.status.state.value,
        }

        if task.status.message is not None:
            parts = task.status.message.parts
            if parts:
                payload["text"] = parts[0].text
                if len(parts) > 1:
                    payload["additional_parts"] = [
                        {"type": part.type, "text": part.text}
                        for part in parts[1:]
                    ]

        # Carry ACP trace_id forward from task metadata if available.
        trace_id_from_metadata = task.metadata.get("acp_trace_id")
        trace_id: str | None = (
            str(trace_id_from_metadata)
            if isinstance(trace_id_from_metadata, str)
            else None
        )

        envelope_metadata: dict[str, str] = {
            "a2a_task_id": task.id,
            "a2a_state": task.status.state.value,
        }

        envelope = MessageEnvelope(
            sender=sender,
            receiver=receiver,
            payload=payload,  # type: ignore[arg-type]
            metadata=envelope_metadata,
        )
        if trace_id is not None:
            envelope.trace_id = trace_id

        return envelope

    def message_to_part(self, payload: dict[str, object]) -> A2AMessage:
        """Wrap an ACP payload dict in an A2A user message.

        Parameters
        ----------
        payload:
            ACP envelope payload dictionary.

        Returns
        -------
        A2AMessage
            A user-role A2A message whose single part contains the payload
            content.  If the payload has a ``"text"`` key, that string is
            used as the part text directly.  Otherwise the entire payload is
            JSON-encoded into the text field.
        """
        if "text" in payload and isinstance(payload["text"], str):
            text_content = payload["text"]
            part_type = "text"
        else:
            try:
                text_content = json.dumps(payload, separators=(",", ":"))
            except (TypeError, ValueError) as exc:
                raise ACPBridgeError(
                    f"Payload cannot be JSON-serialized for A2A conversion: {exc}"
                ) from exc
            part_type = "data"

        part = MessagePart(type=part_type, text=text_content)
        return A2AMessage(role="user", parts=[part])

    def artifact_to_envelope(
        self,
        artifact: Artifact,
        task: A2ATask,
        sender: str,
        receiver: str,
    ) -> MessageEnvelope:
        """Convert a task artifact into an ACP envelope carrying the artifact content.

        Parameters
        ----------
        artifact:
            The artifact to serialize.
        task:
            The parent task (used for metadata).
        sender:
            ACP sender agent ID.
        receiver:
            ACP receiver agent ID.

        Returns
        -------
        MessageEnvelope
            Envelope carrying the artifact's text content and identifying metadata.
        """
        text_parts = [part.text for part in artifact.parts if part.text]
        combined_text = "\n".join(text_parts) if text_parts else ""

        payload: dict[str, object] = {
            "a2a_task_id": task.id,
            "a2a_artifact_name": artifact.name,
            "a2a_artifact_index": artifact.index,
            "text": combined_text,
        }
        metadata: dict[str, str] = {
            "a2a_task_id": task.id,
            "a2a_artifact_name": artifact.name,
        }
        return MessageEnvelope(
            sender=sender,
            receiver=receiver,
            payload=payload,  # type: ignore[arg-type]
            metadata=metadata,
        )
