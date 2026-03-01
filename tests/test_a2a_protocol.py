"""Tests for the A2A v0.3 protocol compatibility layer.

Covers:
- Models: serialization/deserialization round-trips
- AgentCardGenerator: creation, from_config, from_yaml, serialization
- TaskManager: valid/invalid state transitions, history, cancellation
- DiscoveryEndpoint: correct path handling and 404 responses
- ACPBridge: envelope->task, task->envelope, message_to_part, round-trip
- Artifact management
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime

import pytest

from agent_mesh_router.a2a.acp_bridge import ACPBridge, ACPBridgeError
from agent_mesh_router.a2a.agent_card import AgentCardError, AgentCardGenerator
from agent_mesh_router.a2a.discovery import DiscoveryEndpoint
from agent_mesh_router.a2a.models import (
    A2AMessage,
    A2ATask,
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Artifact,
    MessagePart,
    TaskState,
    TaskStatus,
)
from agent_mesh_router.a2a.task_lifecycle import (
    InvalidTaskTransitionError,
    TaskManager,
    TaskNotFoundError,
)
from agent_mesh_router.messages.envelope import MessageEnvelope


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture()
def minimal_card() -> AgentCard:
    return AgentCard(
        name="TestAgent",
        description="A test agent for unit tests.",
        url="https://agents.example.com/test",
    )


@pytest.fixture()
def rich_card() -> AgentCard:
    skill = AgentSkill(
        id="summarize",
        name="Summarize",
        description="Condense text to key points.",
        tags=["nlp", "summarization"],
        examples=["Summarize this article in 3 sentences."],
    )
    caps = AgentCapabilities(
        streaming=True,
        push_notifications=False,
        state_transition_history=True,
    )
    return AgentCard(
        name="SummaryAgent",
        description="Summarises long documents.",
        url="https://agents.example.com/summary",
        version="0.3",
        capabilities=caps,
        skills=[skill],
        default_input_modes=["text/plain", "text/markdown"],
        default_output_modes=["text/plain"],
    )


@pytest.fixture()
def task_manager() -> TaskManager:
    return TaskManager()


@pytest.fixture()
def user_message() -> A2AMessage:
    return A2AMessage(role="user", parts=[MessagePart(text="Hello, agent!")])


@pytest.fixture()
def submitted_task(task_manager: TaskManager, user_message: A2AMessage) -> A2ATask:
    return task_manager.create_task(user_message)


@pytest.fixture()
def bridge() -> ACPBridge:
    return ACPBridge()


@pytest.fixture()
def sample_envelope() -> MessageEnvelope:
    return MessageEnvelope(
        sender="agent-alpha",
        receiver="agent-beta",
        payload={"text": "Process this request."},
    )


# ===========================================================================
# Models — serialization / deserialization round-trips (5 tests)
# ===========================================================================


class TestModelsRoundTrip:
    def test_agent_skill_round_trip(self) -> None:
        skill = AgentSkill(
            id="code-review",
            name="Code Review",
            description="Reviews Python code for issues.",
            tags=["python", "code-quality"],
            examples=["Review this function for bugs."],
        )
        dumped = skill.model_dump_json()
        restored = AgentSkill.model_validate_json(dumped)
        assert restored.id == skill.id
        assert restored.name == skill.name
        assert restored.tags == skill.tags
        assert restored.examples == skill.examples

    def test_agent_card_round_trip(self, rich_card: AgentCard) -> None:
        dumped = rich_card.model_dump_json()
        restored = AgentCard.model_validate_json(dumped)
        assert restored.name == rich_card.name
        assert restored.url == rich_card.url
        assert len(restored.skills) == len(rich_card.skills)
        assert restored.capabilities.streaming is True

    def test_a2a_message_round_trip(self, user_message: A2AMessage) -> None:
        dumped = user_message.model_dump_json()
        restored = A2AMessage.model_validate_json(dumped)
        assert restored.role == "user"
        assert restored.parts[0].text == "Hello, agent!"

    def test_task_status_round_trip(self) -> None:
        status = TaskStatus(
            state=TaskState.WORKING,
            message=A2AMessage(role="agent", parts=[MessagePart(text="In progress")]),
        )
        dumped = status.model_dump_json()
        restored = TaskStatus.model_validate_json(dumped)
        assert restored.state == TaskState.WORKING
        assert restored.message is not None
        assert restored.message.parts[0].text == "In progress"

    def test_a2a_task_round_trip(self, submitted_task: A2ATask) -> None:
        dumped = submitted_task.model_dump_json()
        restored = A2ATask.model_validate_json(dumped)
        assert restored.id == submitted_task.id
        assert restored.status.state == TaskState.SUBMITTED
        assert restored.artifacts == []
        assert restored.history == []


# ===========================================================================
# AgentCardGenerator (8 tests)
# ===========================================================================


class TestAgentCardGenerator:
    def test_create_minimal_card(self) -> None:
        gen = AgentCardGenerator()
        card = gen.from_config(
            {"name": "MinAgent", "description": "Minimal.", "url": "http://localhost"}
        )
        assert card.name == "MinAgent"
        assert card.version == "0.3"
        assert card.capabilities.streaming is False
        assert card.skills == []

    def test_create_card_with_skills(self) -> None:
        gen = AgentCardGenerator()
        config: dict[str, object] = {
            "name": "SkillAgent",
            "description": "Has skills.",
            "url": "http://localhost",
            "skills": [
                {
                    "id": "skill-1",
                    "name": "Skill One",
                    "description": "Does something.",
                    "tags": ["tag-a"],
                }
            ],
        }
        card = gen.from_config(config)
        assert len(card.skills) == 1
        assert card.skills[0].id == "skill-1"
        assert card.skills[0].tags == ["tag-a"]

    def test_create_card_with_capabilities(self) -> None:
        gen = AgentCardGenerator()
        config: dict[str, object] = {
            "name": "CapAgent",
            "description": "Has capabilities.",
            "url": "http://localhost",
            "capabilities": {"streaming": True, "push_notifications": True},
        }
        card = gen.from_config(config)
        assert card.capabilities.streaming is True
        assert card.capabilities.push_notifications is True
        assert card.capabilities.state_transition_history is False

    def test_from_config_missing_name_raises(self) -> None:
        gen = AgentCardGenerator()
        with pytest.raises(AgentCardError, match="name"):
            gen.from_config({"description": "No name.", "url": "http://localhost"})

    def test_from_config_missing_description_raises(self) -> None:
        gen = AgentCardGenerator()
        with pytest.raises(AgentCardError, match="description"):
            gen.from_config({"name": "X", "url": "http://localhost"})

    def test_from_config_missing_url_raises(self) -> None:
        gen = AgentCardGenerator()
        with pytest.raises(AgentCardError, match="url"):
            gen.from_config({"name": "X", "description": "No URL."})

    def test_to_json_produces_valid_json(self, rich_card: AgentCard) -> None:
        gen = AgentCardGenerator()
        raw_json = gen.to_json(rich_card)
        parsed = json.loads(raw_json)
        assert parsed["name"] == "SummaryAgent"
        assert "skills" in parsed
        assert isinstance(parsed["skills"], list)

    def test_from_yaml_round_trip(self) -> None:
        gen = AgentCardGenerator()
        yaml_content = (
            "name: YAMLAgent\n"
            "description: Built from YAML.\n"
            "url: https://agents.example.com/yaml\n"
            "version: '0.3'\n"
            "capabilities:\n"
            "  streaming: true\n"
            "skills:\n"
            "  - id: parse\n"
            "    name: Parse\n"
            "    description: Parses structured data.\n"
            "    tags:\n"
            "      - data\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(yaml_content)
            tmp_path = tmp.name

        try:
            card = gen.from_yaml(tmp_path)
            assert card.name == "YAMLAgent"
            assert card.capabilities.streaming is True
            assert len(card.skills) == 1
            assert card.skills[0].id == "parse"
        finally:
            os.unlink(tmp_path)


# ===========================================================================
# TaskManager state machine (15 tests)
# ===========================================================================


class TestTaskManagerValidTransitions:
    def test_submitted_to_working(
        self, task_manager: TaskManager, submitted_task: A2ATask
    ) -> None:
        task = task_manager.transition(submitted_task.id, TaskState.WORKING)
        assert task.status.state == TaskState.WORKING

    def test_working_to_completed(
        self, task_manager: TaskManager, submitted_task: A2ATask
    ) -> None:
        task_manager.transition(submitted_task.id, TaskState.WORKING)
        task = task_manager.transition(submitted_task.id, TaskState.COMPLETED)
        assert task.status.state == TaskState.COMPLETED

    def test_submitted_working_completed_full_path(
        self, task_manager: TaskManager, user_message: A2AMessage
    ) -> None:
        task = task_manager.create_task(user_message)
        task = task_manager.transition(task.id, TaskState.WORKING)
        task = task_manager.transition(task.id, TaskState.COMPLETED)
        assert task.status.state == TaskState.COMPLETED
        assert task_manager.is_terminal(task.id) is True

    def test_working_to_failed(
        self, task_manager: TaskManager, submitted_task: A2ATask
    ) -> None:
        task_manager.transition(submitted_task.id, TaskState.WORKING)
        task = task_manager.transition(submitted_task.id, TaskState.FAILED)
        assert task.status.state == TaskState.FAILED

    def test_working_to_input_required(
        self, task_manager: TaskManager, submitted_task: A2ATask
    ) -> None:
        task_manager.transition(submitted_task.id, TaskState.WORKING)
        task = task_manager.transition(submitted_task.id, TaskState.INPUT_REQUIRED)
        assert task.status.state == TaskState.INPUT_REQUIRED

    def test_input_required_back_to_working(
        self, task_manager: TaskManager, submitted_task: A2ATask
    ) -> None:
        task_manager.transition(submitted_task.id, TaskState.WORKING)
        task_manager.transition(submitted_task.id, TaskState.INPUT_REQUIRED)
        task = task_manager.transition(submitted_task.id, TaskState.WORKING)
        assert task.status.state == TaskState.WORKING


class TestTaskManagerInvalidTransitions:
    def test_completed_to_working_raises(
        self, task_manager: TaskManager, submitted_task: A2ATask
    ) -> None:
        task_manager.transition(submitted_task.id, TaskState.WORKING)
        task_manager.transition(submitted_task.id, TaskState.COMPLETED)
        with pytest.raises(InvalidTaskTransitionError) as exc_info:
            task_manager.transition(submitted_task.id, TaskState.WORKING)
        assert exc_info.value.current_state == TaskState.COMPLETED
        assert exc_info.value.requested_state == TaskState.WORKING

    def test_submitted_to_completed_raises(
        self, task_manager: TaskManager, submitted_task: A2ATask
    ) -> None:
        with pytest.raises(InvalidTaskTransitionError) as exc_info:
            task_manager.transition(submitted_task.id, TaskState.COMPLETED)
        assert exc_info.value.current_state == TaskState.SUBMITTED

    def test_failed_to_working_raises(
        self, task_manager: TaskManager, submitted_task: A2ATask
    ) -> None:
        task_manager.transition(submitted_task.id, TaskState.WORKING)
        task_manager.transition(submitted_task.id, TaskState.FAILED)
        with pytest.raises(InvalidTaskTransitionError):
            task_manager.transition(submitted_task.id, TaskState.WORKING)


class TestTaskManagerCancellation:
    def test_cancel_from_submitted(
        self, task_manager: TaskManager, submitted_task: A2ATask
    ) -> None:
        task = task_manager.cancel_task(submitted_task.id)
        assert task.status.state == TaskState.CANCELED

    def test_cancel_from_working(
        self, task_manager: TaskManager, submitted_task: A2ATask
    ) -> None:
        task_manager.transition(submitted_task.id, TaskState.WORKING)
        task = task_manager.cancel_task(submitted_task.id)
        assert task.status.state == TaskState.CANCELED

    def test_cancel_from_input_required(
        self, task_manager: TaskManager, submitted_task: A2ATask
    ) -> None:
        task_manager.transition(submitted_task.id, TaskState.WORKING)
        task_manager.transition(submitted_task.id, TaskState.INPUT_REQUIRED)
        task = task_manager.cancel_task(submitted_task.id)
        assert task.status.state == TaskState.CANCELED

    def test_cancel_already_canceled_raises(
        self, task_manager: TaskManager, submitted_task: A2ATask
    ) -> None:
        task_manager.cancel_task(submitted_task.id)
        with pytest.raises(InvalidTaskTransitionError):
            task_manager.cancel_task(submitted_task.id)

    def test_cancel_completed_raises(
        self, task_manager: TaskManager, submitted_task: A2ATask
    ) -> None:
        task_manager.transition(submitted_task.id, TaskState.WORKING)
        task_manager.transition(submitted_task.id, TaskState.COMPLETED)
        with pytest.raises(InvalidTaskTransitionError):
            task_manager.cancel_task(submitted_task.id)


class TestTaskManagerHistory:
    def test_history_recorded_on_transition(
        self, task_manager: TaskManager, submitted_task: A2ATask
    ) -> None:
        task_manager.transition(submitted_task.id, TaskState.WORKING)
        task = task_manager.get_task(submitted_task.id)
        assert len(task.history) == 1
        assert task.history[0].state == TaskState.SUBMITTED

    def test_history_accumulates_across_multiple_transitions(
        self, task_manager: TaskManager, submitted_task: A2ATask
    ) -> None:
        task_manager.transition(submitted_task.id, TaskState.WORKING)
        task_manager.transition(submitted_task.id, TaskState.INPUT_REQUIRED)
        task_manager.transition(submitted_task.id, TaskState.WORKING)
        task = task_manager.transition(submitted_task.id, TaskState.COMPLETED)
        # History should have submitted, working, input-required, working
        assert len(task.history) == 4
        assert task.history[0].state == TaskState.SUBMITTED
        assert task.history[1].state == TaskState.WORKING
        assert task.history[2].state == TaskState.INPUT_REQUIRED
        assert task.history[3].state == TaskState.WORKING

    def test_task_not_found_raises(self, task_manager: TaskManager) -> None:
        with pytest.raises(TaskNotFoundError) as exc_info:
            task_manager.get_task("nonexistent-id")
        assert "nonexistent-id" in str(exc_info.value)


# ===========================================================================
# DiscoveryEndpoint (3 tests)
# ===========================================================================


class TestDiscoveryEndpoint:
    def test_serves_card_at_well_known_path(self, minimal_card: AgentCard) -> None:
        endpoint = DiscoveryEndpoint(minimal_card)
        status, headers, body = endpoint.handle_request("/.well-known/agent.json")
        assert status == 200
        assert headers.get("Content-Type") == "application/json"
        parsed = json.loads(body)
        assert parsed["name"] == "TestAgent"

    def test_returns_404_for_other_paths(self, minimal_card: AgentCard) -> None:
        endpoint = DiscoveryEndpoint(minimal_card)
        for path in ["/health", "/", "/agent.json", "/.well-known/other.json"]:
            status, headers, body = endpoint.handle_request(path)
            assert status == 404, f"Expected 404 for {path}, got {status}"
            assert body == ""

    def test_query_string_stripped_from_path(self, minimal_card: AgentCard) -> None:
        endpoint = DiscoveryEndpoint(minimal_card)
        status, _, body = endpoint.handle_request(
            "/.well-known/agent.json?format=json"
        )
        assert status == 200
        parsed = json.loads(body)
        assert parsed["name"] == "TestAgent"


# ===========================================================================
# ACPBridge (6 tests)
# ===========================================================================


class TestACPBridge:
    def test_envelope_to_task_basic(
        self, bridge: ACPBridge, sample_envelope: MessageEnvelope
    ) -> None:
        task = bridge.envelope_to_task(sample_envelope)
        assert task.status.state == TaskState.SUBMITTED
        assert task.metadata["acp_sender"] == "agent-alpha"
        assert task.metadata["acp_receiver"] == "agent-beta"
        assert task.metadata["acp_trace_id"] == sample_envelope.trace_id

    def test_envelope_to_task_carries_text(
        self, bridge: ACPBridge, sample_envelope: MessageEnvelope
    ) -> None:
        task = bridge.envelope_to_task(sample_envelope)
        assert task.status.message is not None
        assert task.status.message.parts[0].text == "Process this request."
        assert task.status.message.role == "user"

    def test_task_to_envelope_basic(
        self, bridge: ACPBridge, task_manager: TaskManager, user_message: A2AMessage
    ) -> None:
        task = task_manager.create_task(user_message)
        task_manager.transition(task.id, TaskState.WORKING)
        agent_message = A2AMessage(
            role="agent", parts=[MessagePart(text="Working on it.")]
        )
        task_manager.transition(task.id, TaskState.COMPLETED, agent_message)

        envelope = bridge.task_to_envelope(task, sender="agent-beta", receiver="agent-alpha")
        assert envelope.sender == "agent-beta"
        assert envelope.receiver == "agent-alpha"
        assert envelope.payload["a2a_task_id"] == task.id
        assert envelope.payload["a2a_state"] == TaskState.COMPLETED.value

    def test_task_to_envelope_empty_sender_raises(
        self, bridge: ACPBridge, task_manager: TaskManager, user_message: A2AMessage
    ) -> None:
        task = task_manager.create_task(user_message)
        with pytest.raises(ACPBridgeError, match="sender"):
            bridge.task_to_envelope(task, sender="", receiver="agent-alpha")

    def test_message_to_part_with_text_key(self, bridge: ACPBridge) -> None:
        payload: dict[str, object] = {"text": "Hello world"}
        message = bridge.message_to_part(payload)
        assert message.parts[0].text == "Hello world"
        assert message.parts[0].type == "text"

    def test_message_to_part_without_text_key_encodes_json(
        self, bridge: ACPBridge
    ) -> None:
        payload: dict[str, object] = {"action": "summarize", "max_length": 100}
        message = bridge.message_to_part(payload)
        assert message.parts[0].type == "data"
        decoded = json.loads(message.parts[0].text)
        assert decoded["action"] == "summarize"

    def test_round_trip_envelope_to_task_and_back(
        self, bridge: ACPBridge, sample_envelope: MessageEnvelope
    ) -> None:
        task = bridge.envelope_to_task(sample_envelope)
        reply_envelope = bridge.task_to_envelope(
            task, sender="agent-beta", receiver="agent-alpha"
        )
        assert reply_envelope.payload["a2a_task_id"] == task.id
        assert reply_envelope.payload["a2a_state"] == TaskState.SUBMITTED.value


# ===========================================================================
# Artifact management (3 tests)
# ===========================================================================


class TestArtifactManagement:
    def test_add_artifact_to_task(
        self, task_manager: TaskManager, submitted_task: A2ATask
    ) -> None:
        artifact = Artifact(
            name="summary",
            description="A text summary.",
            parts=[MessagePart(text="The document discusses AI safety.")],
            index=0,
        )
        task = task_manager.add_artifact(submitted_task.id, artifact)
        assert len(task.artifacts) == 1
        assert task.artifacts[0].name == "summary"

    def test_add_multiple_artifacts(
        self, task_manager: TaskManager, submitted_task: A2ATask
    ) -> None:
        for i in range(3):
            artifact = Artifact(
                name=f"chunk-{i}",
                description=f"Chunk {i}.",
                parts=[MessagePart(text=f"Content {i}.")],
                index=i,
            )
            task_manager.add_artifact(submitted_task.id, artifact)
        task = task_manager.get_task(submitted_task.id)
        assert len(task.artifacts) == 3
        assert task.artifacts[2].name == "chunk-2"

    def test_artifact_not_found_raises_on_invalid_task(
        self, task_manager: TaskManager
    ) -> None:
        artifact = Artifact(
            name="orphan",
            parts=[MessagePart(text="Lost artifact.")],
        )
        with pytest.raises(TaskNotFoundError):
            task_manager.add_artifact("no-such-task", artifact)


# ===========================================================================
# Additional edge cases (to exceed 60 tests)
# ===========================================================================


class TestAdditionalEdgeCases:
    def test_task_state_enum_values(self) -> None:
        assert TaskState.SUBMITTED.value == "submitted"
        assert TaskState.WORKING.value == "working"
        assert TaskState.INPUT_REQUIRED.value == "input-required"
        assert TaskState.COMPLETED.value == "completed"
        assert TaskState.FAILED.value == "failed"
        assert TaskState.CANCELED.value == "canceled"

    def test_agent_capabilities_defaults(self) -> None:
        caps = AgentCapabilities()
        assert caps.streaming is False
        assert caps.push_notifications is False
        assert caps.state_transition_history is False

    def test_agent_card_default_io_modes(self) -> None:
        card = AgentCard(name="X", description="Y", url="http://z")
        assert card.default_input_modes == ["text/plain"]
        assert card.default_output_modes == ["text/plain"]

    def test_task_manager_no_history_mode(self, user_message: A2AMessage) -> None:
        manager = TaskManager(track_history=False)
        task = manager.create_task(user_message)
        manager.transition(task.id, TaskState.WORKING)
        task = manager.get_task(task.id)
        assert task.history == []

    def test_task_id_is_uuid_string(
        self, submitted_task: A2ATask
    ) -> None:
        import uuid
        # Should not raise
        parsed = uuid.UUID(submitted_task.id)
        assert str(parsed) == submitted_task.id

    def test_task_manager_list_tasks(
        self, task_manager: TaskManager, user_message: A2AMessage
    ) -> None:
        task_manager.create_task(user_message)
        task_manager.create_task(user_message)
        all_tasks = task_manager.list_tasks()
        assert len(all_tasks) == 2

    def test_discovery_endpoint_cache_invalidation(
        self, minimal_card: AgentCard
    ) -> None:
        endpoint = DiscoveryEndpoint(minimal_card)
        # Warm cache
        endpoint.handle_request("/.well-known/agent.json")
        # Invalidate and re-request
        endpoint.invalidate_cache()
        status, _, body = endpoint.handle_request("/.well-known/agent.json")
        assert status == 200
        assert "TestAgent" in body

    def test_agent_card_generator_custom_input_output_modes(self) -> None:
        gen = AgentCardGenerator()
        config: dict[str, object] = {
            "name": "DataAgent",
            "description": "Handles structured data.",
            "url": "http://localhost",
            "default_input_modes": ["application/json", "text/csv"],
            "default_output_modes": ["application/json"],
        }
        card = gen.from_config(config)
        assert "application/json" in card.default_input_modes
        assert "text/csv" in card.default_input_modes
        assert card.default_output_modes == ["application/json"]

    def test_from_yaml_nonexistent_file_raises(self) -> None:
        gen = AgentCardGenerator()
        with pytest.raises(AgentCardError, match="Failed to read"):
            gen.from_yaml("/nonexistent/path/agent.yaml")

    def test_bridge_task_to_envelope_preserves_trace_id(
        self, bridge: ACPBridge, sample_envelope: MessageEnvelope
    ) -> None:
        task = bridge.envelope_to_task(sample_envelope)
        envelope = bridge.task_to_envelope(task, sender="b", receiver="a")
        assert envelope.trace_id == sample_envelope.trace_id

    def test_task_transition_with_message(
        self, task_manager: TaskManager, submitted_task: A2ATask
    ) -> None:
        agent_msg = A2AMessage(
            role="agent",
            parts=[MessagePart(text="I am working.")],
        )
        task = task_manager.transition(
            submitted_task.id, TaskState.WORKING, agent_msg
        )
        assert task.status.message is not None
        assert task.status.message.parts[0].text == "I am working."

    def test_message_part_defaults(self) -> None:
        part = MessagePart()
        assert part.type == "text"
        assert part.text == ""

    def test_artifact_index_default(self) -> None:
        artifact = Artifact(
            name="result",
            parts=[MessagePart(text="Output.")],
        )
        assert artifact.index == 0
        assert artifact.description == ""

    def test_agent_skill_default_tags_and_examples(self) -> None:
        skill = AgentSkill(id="s1", name="Skill 1", description="Does something.")
        assert skill.tags == []
        assert skill.examples == []

    def test_a2a_task_default_fields(self, user_message: A2AMessage) -> None:
        status = TaskStatus(state=TaskState.SUBMITTED, message=user_message)
        task = A2ATask(status=status)
        assert task.artifacts == []
        assert task.history == []
        assert task.metadata == {}
        assert len(task.id) > 0
