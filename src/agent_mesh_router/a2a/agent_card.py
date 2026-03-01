"""AgentCardGenerator — build A2A Agent Cards from AumOS config.

Provides helpers for constructing ``AgentCard`` instances from Python dicts
(typically loaded from YAML or environment config) and serializing them to
the JSON format required by the A2A ``/.well-known/agent.json`` discovery
endpoint.
"""
from __future__ import annotations

import json

from agent_mesh_router.a2a.models import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

# Keys expected in a raw YAML/dict config block for a skill entry.
_SKILL_REQUIRED_KEYS: frozenset[str] = frozenset({"id", "name", "description"})


class AgentCardError(ValueError):
    """Raised when a config dict cannot be used to build a valid AgentCard."""


class AgentCardGenerator:
    """Builds A2A Agent Cards from AumOS agent config.

    All methods are stateless — the generator can be shared across threads
    without locking.

    Example
    -------
    ::

        generator = AgentCardGenerator()
        card = generator.from_config({
            "name": "SummaryAgent",
            "description": "Summarises long documents",
            "url": "https://agents.example.com/summary",
            "skills": [
                {
                    "id": "summarize",
                    "name": "Summarize",
                    "description": "Condense text to key points",
                    "tags": ["nlp", "summarization"],
                }
            ],
        })
        payload = generator.to_json(card)
    """

    def from_config(self, config: dict[str, object]) -> AgentCard:
        """Generate an Agent Card from an AumOS config dict.

        Parameters
        ----------
        config:
            Dictionary with at minimum ``name``, ``description``, and ``url``
            keys.  Optional keys: ``version``, ``capabilities``, ``skills``,
            ``default_input_modes``, ``default_output_modes``.

        Returns
        -------
        AgentCard
            Fully populated Agent Card.

        Raises
        ------
        AgentCardError
            If required keys are missing or have invalid types.
        """
        for required_key in ("name", "description", "url"):
            if required_key not in config:
                raise AgentCardError(
                    f"AgentCard config is missing required key: {required_key!r}."
                )
            if not isinstance(config[required_key], str) or not config[required_key]:
                raise AgentCardError(
                    f"AgentCard config key {required_key!r} must be a non-empty string."
                )

        capabilities = self._parse_capabilities(config.get("capabilities"))
        skills = self._parse_skills(config.get("skills", []))

        raw_input_modes = config.get("default_input_modes", ["text/plain"])
        raw_output_modes = config.get("default_output_modes", ["text/plain"])

        input_modes: list[str] = (
            list(raw_input_modes)
            if isinstance(raw_input_modes, list)
            else ["text/plain"]
        )
        output_modes: list[str] = (
            list(raw_output_modes)
            if isinstance(raw_output_modes, list)
            else ["text/plain"]
        )

        version = str(config.get("version", "0.3"))

        return AgentCard(
            name=str(config["name"]),
            description=str(config["description"]),
            url=str(config["url"]),
            version=version,
            capabilities=capabilities,
            skills=skills,
            default_input_modes=input_modes,
            default_output_modes=output_modes,
        )

    def from_yaml(self, yaml_path: str) -> AgentCard:
        """Generate an Agent Card by reading a YAML file.

        Parameters
        ----------
        yaml_path:
            Filesystem path to a YAML file whose top-level keys match the
            ``from_config`` dict schema.

        Returns
        -------
        AgentCard
            Fully populated Agent Card.

        Raises
        ------
        AgentCardError
            If the file cannot be read, is not valid YAML, or the resulting
            dict is missing required keys.
        FileNotFoundError
            If ``yaml_path`` does not point to an existing file.
        """
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise AgentCardError(
                "PyYAML is required for from_yaml(). Install it with: pip install pyyaml"
            ) from exc

        try:
            with open(yaml_path, encoding="utf-8") as file_handle:
                raw_config: object = yaml.safe_load(file_handle)
        except OSError as exc:
            raise AgentCardError(
                f"Failed to read Agent Card YAML file at {yaml_path!r}: {exc}"
            ) from exc
        except yaml.YAMLError as exc:
            raise AgentCardError(
                f"Invalid YAML in Agent Card file at {yaml_path!r}: {exc}"
            ) from exc

        if not isinstance(raw_config, dict):
            raise AgentCardError(
                f"Agent Card YAML must be a mapping, got {type(raw_config).__name__}."
            )

        return self.from_config(raw_config)

    def to_json(self, card: AgentCard) -> str:
        """Serialize an Agent Card to JSON for serving at /.well-known/agent.json.

        Parameters
        ----------
        card:
            The Agent Card to serialize.

        Returns
        -------
        str
            JSON string suitable for use as an HTTP response body.
        """
        return card.model_dump_json(indent=2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_capabilities(
        self, raw: object
    ) -> AgentCapabilities:
        """Parse a raw capabilities block into an AgentCapabilities model."""
        if raw is None:
            return AgentCapabilities()
        if not isinstance(raw, dict):
            return AgentCapabilities()
        return AgentCapabilities(
            streaming=bool(raw.get("streaming", False)),
            push_notifications=bool(raw.get("push_notifications", False)),
            state_transition_history=bool(raw.get("state_transition_history", False)),
        )

    def _parse_skills(self, raw: object) -> list[AgentSkill]:
        """Parse a raw skills list into a list of AgentSkill models."""
        if not isinstance(raw, list):
            return []
        skills: list[AgentSkill] = []
        for index, skill_raw in enumerate(raw):
            if not isinstance(skill_raw, dict):
                raise AgentCardError(
                    f"Skill at index {index} must be a mapping, "
                    f"got {type(skill_raw).__name__}."
                )
            missing = _SKILL_REQUIRED_KEYS - skill_raw.keys()
            if missing:
                raise AgentCardError(
                    f"Skill at index {index} is missing required keys: "
                    f"{sorted(missing)}."
                )
            tags_raw = skill_raw.get("tags", [])
            examples_raw = skill_raw.get("examples", [])
            skills.append(
                AgentSkill(
                    id=str(skill_raw["id"]),
                    name=str(skill_raw["name"]),
                    description=str(skill_raw["description"]),
                    tags=list(tags_raw) if isinstance(tags_raw, list) else [],
                    examples=list(examples_raw) if isinstance(examples_raw, list) else [],
                )
            )
        return skills
