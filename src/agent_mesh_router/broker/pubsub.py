"""TopicManager — wildcard pub/sub topic routing for the agent mesh.

Provides a standalone topic manager that supports wildcard topic matching
using the ``*`` glob character.  Patterns like ``agent.*`` match any
single-segment topic under the ``agent`` namespace (e.g. ``agent.status``,
``agent.heartbeat``), while ``*`` matches any single-segment topic and
``agent.*.status`` matches any intermediate segment.

Design notes
------------
- Subscription maps store (agent_id, pattern) pairs so the same agent can
  subscribe to multiple patterns independently.
- Pattern matching uses ``fnmatch.fnmatchcase`` which handles ``*`` and
  ``?`` wildcards with no regex overhead.
- All public methods are synchronous to keep this layer dependency-free;
  async callers should run these in the default executor if needed.
- Thread-safe via a single ``threading.RLock``.

Example
-------
::

    from agent_mesh_router.broker.pubsub import TopicManager

    manager = TopicManager()
    manager.subscribe("agent-a", "events.*")
    manager.subscribe("agent-b", "events.error")

    delivered = manager.publish("events.error", {"code": 500})
    # delivered == {"agent-a", "agent-b"}

    manager.unsubscribe("agent-a", "events.*")
    delivered = manager.publish("events.error", {"code": 500})
    # delivered == {"agent-b"}
"""
from __future__ import annotations

import fnmatch
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TopicSubscription:
    """A single (agent_id, pattern) subscription entry.

    Attributes
    ----------
    agent_id:
        The subscribing agent's unique identifier.
    pattern:
        Topic pattern string; may include ``*`` wildcards.
        Exact strings are also valid and match only themselves.
    subscribed_at:
        Unix epoch seconds when the subscription was created.
    """

    agent_id: str
    pattern: str
    subscribed_at: float = field(default_factory=lambda: __import__("time").time())


class TopicManager:
    """Thread-safe topic subscription manager with wildcard matching.

    Agents subscribe to *patterns* (which may be exact topic strings or
    glob-style patterns using ``*``).  When a message is published to a
    concrete topic, all agents whose patterns match that topic are returned
    as delivery targets.

    Parameters
    ----------
    case_sensitive:
        When True (default), topic and pattern matching is case-sensitive.
        When False, both topics and patterns are lowercased before matching.

    Example
    -------
    ::

        manager = TopicManager()
        manager.subscribe("agent-a", "metrics.*")
        manager.subscribe("agent-b", "metrics.cpu")

        # Both agents receive "metrics.cpu"
        targets = manager.publish("metrics.cpu", {"value": 0.82})
        assert targets == {"agent-a", "agent-b"}

        # Only agent-a's wildcard matches "metrics.memory"
        targets = manager.publish("metrics.memory", {"value": 0.45})
        assert targets == {"agent-a"}
    """

    def __init__(self, *, case_sensitive: bool = True) -> None:
        self._case_sensitive = case_sensitive
        # pattern → set[agent_id]
        self._subscriptions: dict[str, set[str]] = defaultdict(set)
        # agent_id → set[pattern] — inverse index for fast unsubscription
        self._agent_patterns: dict[str, set[str]] = defaultdict(set)
        self._lock = threading.RLock()
        self._publish_count: int = 0
        self._total_deliveries: int = 0

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    def subscribe(self, agent_id: str, pattern: str) -> None:
        """Subscribe ``agent_id`` to all topics matching ``pattern``.

        Re-subscribing the same (agent_id, pattern) pair is a no-op.

        Parameters
        ----------
        agent_id:
            The agent wishing to receive messages.
        pattern:
            Topic pattern; may contain ``*`` wildcards (e.g. ``"agent.*"``).
            An exact topic string is also accepted.

        Raises
        ------
        ValueError
            If ``agent_id`` or ``pattern`` is an empty string.
        """
        if not agent_id or not agent_id.strip():
            raise ValueError("agent_id must be a non-empty string.")
        if not pattern or not pattern.strip():
            raise ValueError("pattern must be a non-empty string.")

        normalized_pattern = self._normalize(pattern)
        with self._lock:
            self._subscriptions[normalized_pattern].add(agent_id)
            self._agent_patterns[agent_id].add(normalized_pattern)

        logger.debug("Agent %r subscribed to pattern %r.", agent_id, pattern)

    def unsubscribe(self, agent_id: str, pattern: str) -> None:
        """Remove the (agent_id, pattern) subscription.

        Silently does nothing if the subscription does not exist.

        Parameters
        ----------
        agent_id:
            The agent to remove.
        pattern:
            The exact pattern string used when subscribing.
        """
        normalized_pattern = self._normalize(pattern)
        with self._lock:
            self._subscriptions[normalized_pattern].discard(agent_id)
            if not self._subscriptions[normalized_pattern]:
                del self._subscriptions[normalized_pattern]
            self._agent_patterns[agent_id].discard(normalized_pattern)
            if not self._agent_patterns[agent_id]:
                del self._agent_patterns[agent_id]

        logger.debug("Agent %r unsubscribed from pattern %r.", agent_id, pattern)

    def unsubscribe_all(self, agent_id: str) -> int:
        """Remove all subscriptions for ``agent_id``.

        Parameters
        ----------
        agent_id:
            The agent to fully remove from the topic manager.

        Returns
        -------
        int
            Number of pattern subscriptions that were removed.
        """
        with self._lock:
            patterns = set(self._agent_patterns.pop(agent_id, set()))
            count = len(patterns)
            for pattern in patterns:
                self._subscriptions[pattern].discard(agent_id)
                if not self._subscriptions[pattern]:
                    del self._subscriptions[pattern]

        logger.debug(
            "Agent %r unsubscribed from all %d patterns.", agent_id, count
        )
        return count

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish(
        self,
        topic: str,
        payload: dict[str, object],
    ) -> set[str]:
        """Return the set of agent IDs whose subscribed patterns match ``topic``.

        This method does NOT deliver messages itself — it returns the set
        of matching agent IDs so the caller can dispatch through the broker.

        Parameters
        ----------
        topic:
            The concrete topic string to match against subscribed patterns.
        payload:
            Informational payload (not used internally; passed for logging).

        Returns
        -------
        set[str]
            Agent IDs that should receive this message.  May be empty.
        """
        normalized_topic = self._normalize(topic)
        matched: set[str] = set()

        with self._lock:
            patterns = list(self._subscriptions.keys())

        for pattern in patterns:
            if fnmatch.fnmatchcase(normalized_topic, pattern):
                with self._lock:
                    matched.update(self._subscriptions.get(pattern, set()))

        with self._lock:
            self._publish_count += 1
            self._total_deliveries += len(matched)

        logger.debug(
            "Topic %r matched %d subscriber(s): %r",
            topic,
            len(matched),
            matched,
        )
        return matched

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def list_topics(self) -> list[str]:
        """Return a sorted list of all active subscription patterns.

        Note: these are the *patterns* agents have subscribed to, not
        the concrete topics that have been published.

        Returns
        -------
        list[str]
            Sorted list of active pattern strings.
        """
        with self._lock:
            return sorted(self._subscriptions.keys())

    def list_subscribers(self, pattern: str) -> set[str]:
        """Return the set of agents subscribed to an exact pattern.

        Parameters
        ----------
        pattern:
            The exact pattern string (not a topic — must match how the
            subscription was originally registered).

        Returns
        -------
        set[str]
            Copy of the subscriber set; empty if pattern has no subscribers.
        """
        normalized_pattern = self._normalize(pattern)
        with self._lock:
            return set(self._subscriptions.get(normalized_pattern, set()))

    def list_agent_patterns(self, agent_id: str) -> set[str]:
        """Return all patterns that ``agent_id`` is subscribed to.

        Parameters
        ----------
        agent_id:
            The agent to query.

        Returns
        -------
        set[str]
            Copy of the agent's pattern set; empty if not subscribed to anything.
        """
        with self._lock:
            return set(self._agent_patterns.get(agent_id, set()))

    def resolve_subscribers(self, topic: str) -> set[str]:
        """Return agent IDs matching ``topic`` without incrementing publish counters.

        Useful for dry-run inspection.

        Parameters
        ----------
        topic:
            Concrete topic to test against all registered patterns.

        Returns
        -------
        set[str]
            Matched agent IDs.
        """
        normalized_topic = self._normalize(topic)
        matched: set[str] = set()

        with self._lock:
            for pattern, subscribers in self._subscriptions.items():
                if fnmatch.fnmatchcase(normalized_topic, pattern):
                    matched.update(subscribers)

        return matched

    def subscriber_count(self, pattern: str | None = None) -> int:
        """Return the number of subscriptions.

        Parameters
        ----------
        pattern:
            When provided, returns the count for that specific pattern.
            When None, returns the total unique agent subscription count
            across all patterns.

        Returns
        -------
        int
            Subscription count.
        """
        with self._lock:
            if pattern is not None:
                normalized = self._normalize(pattern)
                return len(self._subscriptions.get(normalized, set()))
            # Total unique agents
            all_agents: set[str] = set()
            for subscribers in self._subscriptions.values():
                all_agents.update(subscribers)
            return len(all_agents)

    def topic_count(self) -> int:
        """Return the number of active subscription patterns.

        Returns
        -------
        int
            Number of distinct patterns with at least one subscriber.
        """
        with self._lock:
            return len(self._subscriptions)

    @property
    def publish_count(self) -> int:
        """Total number of ``publish()`` calls made since construction."""
        return self._publish_count

    @property
    def total_deliveries(self) -> int:
        """Total cumulative agent-delivery hits across all ``publish()`` calls."""
        return self._total_deliveries

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize(self, value: str) -> str:
        """Apply case normalization if configured."""
        return value if self._case_sensitive else value.lower()

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"TopicManager("
                f"patterns={self.topic_count()}, "
                f"publishes={self._publish_count}, "
                f"deliveries={self._total_deliveries}"
                f")"
            )
