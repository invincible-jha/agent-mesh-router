"""Deadlock detection via cycle detection in a directed wait-for graph.

Design
------
Agents declare *wait-for* relationships: "agent A is waiting for agent B
to release a resource".  These relationships form a directed graph.  A
cycle in this graph indicates a deadlock — no agent in the cycle can
make progress without the others.

The detector uses iterative DFS (no recursion limit risk) to enumerate
all simple cycles in the graph.  It reports each cycle as the ordered
sequence of agent IDs involved.

Usage
-----
::

    from agent_mesh_router.deadlock import DeadlockDetector

    detector = DeadlockDetector()
    detector.add_wait("agent_a", "agent_b")  # A waits for B
    detector.add_wait("agent_b", "agent_c")  # B waits for C
    detector.add_wait("agent_c", "agent_a")  # C waits for A  → deadlock!

    result = detector.detect()
    if result.has_deadlock:
        for cycle in result.cycles:
            print("Deadlock cycle:", " -> ".join(cycle))
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WaitEdge:
    """A directed wait-for relationship between two agents.

    Parameters
    ----------
    waiter:
        The agent that is blocked waiting.
    holder:
        The agent that holds the resource the waiter needs.
    resource_id:
        Optional label for the resource being contested.
    """

    waiter: str
    holder: str
    resource_id: str = ""

    def __str__(self) -> str:
        res = f" [{self.resource_id}]" if self.resource_id else ""
        return f"{self.waiter} -waits-for-> {self.holder}{res}"


@dataclass(frozen=True)
class DeadlockResult:
    """Immutable result of a deadlock detection run.

    Parameters
    ----------
    cycles:
        List of detected cycles.  Each cycle is an ordered list of agent
        IDs forming the loop (the first and last element are the same
        agent to make the cycle explicit).
    checked_at:
        UTC timestamp when the detection was run.
    node_count:
        Total number of distinct agents in the graph at detection time.
    edge_count:
        Total number of wait-for edges in the graph at detection time.
    """

    cycles: tuple[tuple[str, ...], ...]
    checked_at: datetime
    node_count: int
    edge_count: int

    @property
    def has_deadlock(self) -> bool:
        """Return True when at least one cycle was found."""
        return len(self.cycles) > 0

    @property
    def cycle_count(self) -> int:
        """Number of distinct cycles detected."""
        return len(self.cycles)

    def summary(self) -> str:
        """Return a human-readable one-line summary."""
        if not self.has_deadlock:
            return (
                f"No deadlock detected. "
                f"Checked {self.node_count} agents, {self.edge_count} edges "
                f"at {self.checked_at.isoformat()}."
            )
        cycle_strs = [" -> ".join(c) for c in self.cycles]
        return (
            f"DEADLOCK: {self.cycle_count} cycle(s) found. "
            + " | ".join(cycle_strs)
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dictionary."""
        return {
            "has_deadlock": self.has_deadlock,
            "cycle_count": self.cycle_count,
            "cycles": [list(c) for c in self.cycles],
            "checked_at": self.checked_at.isoformat(),
            "node_count": self.node_count,
            "edge_count": self.edge_count,
        }


# ---------------------------------------------------------------------------
# DeadlockDetector
# ---------------------------------------------------------------------------


class DeadlockDetector:
    """Detect cycles in a directed agent wait-for graph.

    All public methods are thread-safe (protected by a single lock).

    Parameters
    ----------
    max_cycle_length:
        Stop reporting cycles that exceed this length.  Helps bound output
        for very large graphs.  Defaults to 50.

    Example
    -------
    ::

        detector = DeadlockDetector()
        detector.add_wait("a", "b")
        detector.add_wait("b", "c")
        detector.add_wait("c", "a")
        result = detector.detect()
        assert result.has_deadlock
    """

    def __init__(self, max_cycle_length: int = 50) -> None:
        if max_cycle_length < 2:
            raise ValueError(
                f"max_cycle_length must be >= 2, got {max_cycle_length!r}."
            )
        self._max_cycle_length = max_cycle_length
        # Adjacency list: waiter -> set of holders
        self._graph: dict[str, set[str]] = {}
        self._edges: list[WaitEdge] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_wait(
        self,
        waiter: str,
        holder: str,
        resource_id: str = "",
    ) -> None:
        """Register a wait-for relationship.

        Parameters
        ----------
        waiter:
            Agent that is blocked.
        holder:
            Agent that holds the resource.
        resource_id:
            Optional label for the contested resource.

        Raises
        ------
        ValueError
            If *waiter* equals *holder* (self-loops are always deadlocks
            and are rejected to keep the graph simple).
        """
        if waiter == holder:
            raise ValueError(
                f"Self-loop detected: agent '{waiter}' cannot wait for itself."
            )
        with self._lock:
            if waiter not in self._graph:
                self._graph[waiter] = set()
            if holder not in self._graph:
                self._graph[holder] = set()
            self._graph[waiter].add(holder)
            self._edges.append(WaitEdge(waiter=waiter, holder=holder, resource_id=resource_id))

    def remove_wait(self, waiter: str, holder: str) -> bool:
        """Remove a specific wait-for edge.

        Parameters
        ----------
        waiter:
            The blocked agent.
        holder:
            The holder agent.

        Returns
        -------
        bool
            True if the edge existed and was removed; False otherwise.
        """
        with self._lock:
            if waiter not in self._graph:
                return False
            if holder not in self._graph[waiter]:
                return False
            self._graph[waiter].discard(holder)
            self._edges = [
                e for e in self._edges if not (e.waiter == waiter and e.holder == holder)
            ]
            return True

    def clear_agent(self, agent_id: str) -> None:
        """Remove all edges involving *agent_id* (waiter or holder).

        Parameters
        ----------
        agent_id:
            The agent whose edges should be removed.
        """
        with self._lock:
            # Remove outgoing edges
            self._graph.pop(agent_id, None)
            # Remove incoming edges
            for neighbors in self._graph.values():
                neighbors.discard(agent_id)
            # Remove from edge list
            self._edges = [
                e for e in self._edges
                if e.waiter != agent_id and e.holder != agent_id
            ]

    def clear(self) -> None:
        """Remove all agents and wait edges from the graph."""
        with self._lock:
            self._graph.clear()
            self._edges.clear()

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self) -> DeadlockResult:
        """Run cycle detection and return a :class:`DeadlockResult`.

        Uses iterative DFS to find all simple cycles in the directed graph.
        The algorithm is Johnson's algorithm simplified to a single-pass DFS
        with a path stack — suitable for the graph sizes expected in agent
        meshes (typically tens to hundreds of nodes).

        Returns
        -------
        DeadlockResult
            Contains all detected cycles and summary statistics.
        """
        with self._lock:
            # Snapshot for analysis (avoids holding lock during DFS)
            graph_snapshot: dict[str, list[str]] = {
                node: sorted(neighbors)
                for node, neighbors in self._graph.items()
            }
            node_count = len(graph_snapshot)
            edge_count = sum(len(ns) for ns in graph_snapshot.values())
            checked_at = datetime.now(timezone.utc)

        cycles = list(self._find_all_cycles(graph_snapshot))
        # Deduplicate by canonical form (rotate to smallest node first)
        unique_cycles = _deduplicate_cycles(cycles)

        return DeadlockResult(
            cycles=tuple(unique_cycles),
            checked_at=checked_at,
            node_count=node_count,
            edge_count=edge_count,
        )

    def agents(self) -> list[str]:
        """Return all agent IDs currently in the graph (sorted)."""
        with self._lock:
            return sorted(self._graph.keys())

    def edges(self) -> list[WaitEdge]:
        """Return a copy of all wait edges currently registered."""
        with self._lock:
            return list(self._edges)

    def __len__(self) -> int:
        """Return the number of agents in the graph."""
        with self._lock:
            return len(self._graph)

    def __contains__(self, agent_id: object) -> bool:
        """Return True if *agent_id* is registered."""
        with self._lock:
            return agent_id in self._graph

    # ------------------------------------------------------------------
    # Internal — DFS cycle finder
    # ------------------------------------------------------------------

    def _find_all_cycles(
        self, graph: dict[str, list[str]]
    ) -> Iterator[tuple[str, ...]]:
        """Yield all simple cycles in *graph* using iterative DFS.

        Parameters
        ----------
        graph:
            Adjacency list mapping node -> sorted list of neighbors.
        """
        nodes = list(graph.keys())

        for start in nodes:
            # Stack entries: (current_node, path_so_far, visited_in_path)
            stack: list[tuple[str, list[str], set[str]]] = [
                (start, [start], {start})
            ]
            while stack:
                current, path, visited = stack.pop()

                if len(path) > self._max_cycle_length:
                    continue

                for neighbor in graph.get(current, []):
                    if neighbor == start and len(path) >= 2:
                        # Found a cycle back to start
                        cycle = tuple(path + [start])
                        yield cycle
                    elif neighbor not in visited and neighbor > start:
                        # Only expand to nodes greater than start (lexicographic)
                        # This avoids reporting the same cycle multiple times
                        # from different starting points.
                        stack.append(
                            (neighbor, path + [neighbor], visited | {neighbor})
                        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _canonical_cycle(cycle: tuple[str, ...]) -> tuple[str, ...]:
    """Return the cycle starting at the lexicographically smallest node.

    The last element repeats the first, so we normalise on the path
    elements only (excluding the closing repeat).
    """
    path = cycle[:-1]  # drop the repeated closing node
    if not path:
        return cycle
    min_idx = path.index(min(path))
    rotated = path[min_idx:] + path[:min_idx]
    return tuple(rotated) + (rotated[0],)


def _deduplicate_cycles(
    cycles: list[tuple[str, ...]],
) -> list[tuple[str, ...]]:
    """Return unique cycles in canonical form."""
    seen: set[tuple[str, ...]] = set()
    result: list[tuple[str, ...]] = []
    for cycle in cycles:
        canonical = _canonical_cycle(cycle)
        if canonical not in seen:
            seen.add(canonical)
            result.append(canonical)
    return result
