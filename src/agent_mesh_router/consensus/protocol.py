"""Consensus protocol implementations for multi-agent agreement.

Design
------
All implementations follow the same interface defined by
:class:`ConsensusProtocol`.  Each agent submits a :class:`VoteRecord`
containing its preferred option and optional confidence score.  The
protocol resolves the votes and returns a :class:`ConsensusResult`.

All implementations are synchronous and thread-safe.  They are stateless
between rounds — call :meth:`~ConsensusProtocol.resolve` once per
decision round.

Implementations
---------------
:class:`MajorityVote`
    The option that receives strictly more than half the votes wins.
    Ties are broken by lexicographic order of the option string.
:class:`ConfidenceWeighted`
    Votes are weighted by ``confidence``.  The option with the highest
    aggregate weight wins.  Falls back to MajorityVote when all
    confidences are 0.
:class:`Hierarchical`
    An authority agent's vote is final.  If the authority has not voted,
    or the authority agent is not present in the vote list, the result
    falls back to :class:`MajorityVote`.

Usage
-----
::

    from agent_mesh_router.consensus import MajorityVote, VoteRecord

    protocol = MajorityVote()
    votes = [
        VoteRecord(agent_id="a", option="deploy"),
        VoteRecord(agent_id="b", option="deploy"),
        VoteRecord(agent_id="c", option="rollback"),
    ]
    result = protocol.resolve(votes)
    assert result.winning_option == "deploy"
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# VoteRecord
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VoteRecord:
    """A single agent's vote in a consensus round.

    Parameters
    ----------
    agent_id:
        Unique identifier of the voting agent.
    option:
        The option this agent votes for (e.g. ``"deploy"``, ``"rollback"``).
    confidence:
        Confidence score in [0.0, 1.0].  Used by :class:`ConfidenceWeighted`.
        Defaults to 1.0 (full confidence).
    metadata:
        Arbitrary key-value annotations.
    """

    agent_id: str
    option: str
    confidence: float = 1.0
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.agent_id.strip():
            raise ValueError("agent_id must not be empty.")
        if not self.option.strip():
            raise ValueError("option must not be empty.")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {self.confidence!r}."
            )


# ---------------------------------------------------------------------------
# ConsensusResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsensusResult:
    """Immutable outcome of a consensus round.

    Parameters
    ----------
    winning_option:
        The option that the protocol selected.
    vote_counts:
        Raw vote tallies per option.
    total_votes:
        Total number of votes submitted.
    agreement_ratio:
        Fraction of voters that voted for the winning option (0.0–1.0).
    reached_consensus:
        True when the protocol produced a definitive result.  May be False
        when no votes were submitted.
    protocol_name:
        Name of the protocol that produced this result.
    decided_at:
        UTC timestamp of the decision.
    metadata:
        Additional protocol-specific information.
    """

    winning_option: str
    vote_counts: dict[str, int]
    total_votes: int
    agreement_ratio: float
    reached_consensus: bool
    protocol_name: str
    decided_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a one-line human-readable summary."""
        if not self.reached_consensus:
            return f"[{self.protocol_name}] No consensus — 0 votes submitted."
        return (
            f"[{self.protocol_name}] Consensus: {self.winning_option!r} "
            f"({self.vote_counts.get(self.winning_option, 0)}/{self.total_votes} votes, "
            f"agreement={self.agreement_ratio:.1%})"
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dictionary."""
        return {
            "winning_option": self.winning_option,
            "vote_counts": self.vote_counts,
            "total_votes": self.total_votes,
            "agreement_ratio": self.agreement_ratio,
            "reached_consensus": self.reached_consensus,
            "protocol_name": self.protocol_name,
            "decided_at": self.decided_at.isoformat(),
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class ConsensusProtocol(ABC):
    """Abstract base class for consensus protocols.

    Subclasses must implement :meth:`resolve`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short protocol name (e.g. ``"majority_vote"``)."""

    @abstractmethod
    def resolve(self, votes: list[VoteRecord]) -> ConsensusResult:
        """Resolve votes and return a :class:`ConsensusResult`.

        Parameters
        ----------
        votes:
            List of :class:`VoteRecord` from participating agents.
            May be empty — in which case ``reached_consensus`` will be False.

        Returns
        -------
        ConsensusResult
            The resolved outcome.
        """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _no_consensus_result(protocol_name: str) -> ConsensusResult:
    return ConsensusResult(
        winning_option="",
        vote_counts={},
        total_votes=0,
        agreement_ratio=0.0,
        reached_consensus=False,
        protocol_name=protocol_name,
        decided_at=datetime.now(timezone.utc),
    )


def _tiebreak(options: list[str]) -> str:
    """Return the lexicographically smallest option to break ties."""
    return min(options)


# ---------------------------------------------------------------------------
# MajorityVote
# ---------------------------------------------------------------------------


class MajorityVote(ConsensusProtocol):
    """Simple majority vote consensus.

    The option that receives the most votes wins.  On a tie, the
    lexicographically smallest option is selected.

    Parameters
    ----------
    require_majority:
        If True (default), the winning option must have strictly more than
        50% of votes.  If it does not, ``reached_consensus`` is False.
        If False, plurality is sufficient.

    Example
    -------
    ::

        protocol = MajorityVote()
        votes = [VoteRecord("a", "yes"), VoteRecord("b", "yes"), VoteRecord("c", "no")]
        result = protocol.resolve(votes)
        assert result.winning_option == "yes"
    """

    def __init__(self, require_majority: bool = True) -> None:
        self._require_majority = require_majority
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "majority_vote"

    def resolve(self, votes: list[VoteRecord]) -> ConsensusResult:
        if not votes:
            return _no_consensus_result(self.name)

        with self._lock:
            counts: Counter[str] = Counter(v.option for v in votes)
            total = len(votes)
            max_count = max(counts.values())
            top_options = [opt for opt, cnt in counts.items() if cnt == max_count]
            winning_option = _tiebreak(top_options)

            agreement_ratio = max_count / total
            reached = True
            if self._require_majority and agreement_ratio <= 0.5 and total > 1:
                reached = False

            return ConsensusResult(
                winning_option=winning_option,
                vote_counts=dict(counts),
                total_votes=total,
                agreement_ratio=agreement_ratio,
                reached_consensus=reached,
                protocol_name=self.name,
                decided_at=datetime.now(timezone.utc),
            )


# ---------------------------------------------------------------------------
# ConfidenceWeighted
# ---------------------------------------------------------------------------


class ConfidenceWeighted(ConsensusProtocol):
    """Confidence-weighted consensus.

    Each vote contributes its ``confidence`` score as weight.  The option
    with the highest aggregate weight wins.  When all confidence values are
    0, falls back to plurality (equal weights).

    Parameters
    ----------
    min_winning_weight_ratio:
        The winner's total weight must represent at least this fraction of
        total weight for ``reached_consensus`` to be True.
        Defaults to 0.0 (any win counts).

    Example
    -------
    ::

        protocol = ConfidenceWeighted()
        votes = [
            VoteRecord("a", "yes", confidence=0.9),
            VoteRecord("b", "no", confidence=0.1),
        ]
        result = protocol.resolve(votes)
        assert result.winning_option == "yes"
    """

    def __init__(self, min_winning_weight_ratio: float = 0.0) -> None:
        if not (0.0 <= min_winning_weight_ratio <= 1.0):
            raise ValueError(
                f"min_winning_weight_ratio must be in [0.0, 1.0], "
                f"got {min_winning_weight_ratio!r}."
            )
        self._min_ratio = min_winning_weight_ratio
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "confidence_weighted"

    def resolve(self, votes: list[VoteRecord]) -> ConsensusResult:
        if not votes:
            return _no_consensus_result(self.name)

        with self._lock:
            # Build raw counts and weight maps
            weight_map: dict[str, float] = {}
            count_map: Counter[str] = Counter()
            for vote in votes:
                weight_map[vote.option] = weight_map.get(vote.option, 0.0) + vote.confidence
                count_map[vote.option] += 1

            total_weight = sum(weight_map.values())
            if total_weight == 0.0:
                # Fallback: equal weights (pure count)
                weight_map = {opt: float(cnt) for opt, cnt in count_map.items()}
                total_weight = float(len(votes))

            max_weight = max(weight_map.values())
            top_options = [opt for opt, w in weight_map.items() if w == max_weight]
            winning_option = _tiebreak(top_options)

            agreement_ratio = max_weight / total_weight if total_weight > 0 else 0.0
            reached = agreement_ratio >= self._min_ratio

            return ConsensusResult(
                winning_option=winning_option,
                vote_counts=dict(count_map),
                total_votes=len(votes),
                agreement_ratio=agreement_ratio,
                reached_consensus=reached,
                protocol_name=self.name,
                decided_at=datetime.now(timezone.utc),
                metadata={"total_weight": total_weight, "winning_weight": max_weight},
            )


# ---------------------------------------------------------------------------
# Hierarchical
# ---------------------------------------------------------------------------


class Hierarchical(ConsensusProtocol):
    """Authority-first consensus with majority fallback.

    If the designated authority agent submitted a vote, that vote is the
    final answer regardless of what other agents voted.  If the authority
    did not vote, falls back to :class:`MajorityVote`.

    Parameters
    ----------
    authority_agent_id:
        The agent ID whose vote is treated as authoritative.

    Example
    -------
    ::

        protocol = Hierarchical(authority_agent_id="lead_agent")
        votes = [
            VoteRecord("lead_agent", "deploy"),
            VoteRecord("worker_a", "rollback"),
        ]
        result = protocol.resolve(votes)
        assert result.winning_option == "deploy"  # authority overrides
    """

    def __init__(self, authority_agent_id: str) -> None:
        if not authority_agent_id.strip():
            raise ValueError("authority_agent_id must not be empty.")
        self._authority_id = authority_agent_id
        self._fallback = MajorityVote(require_majority=False)
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "hierarchical"

    @property
    def authority_agent_id(self) -> str:
        """Return the authority agent identifier."""
        return self._authority_id

    def resolve(self, votes: list[VoteRecord]) -> ConsensusResult:
        if not votes:
            return _no_consensus_result(self.name)

        with self._lock:
            # Check if authority voted
            authority_votes = [v for v in votes if v.agent_id == self._authority_id]
            counts: Counter[str] = Counter(v.option for v in votes)
            total = len(votes)

            if authority_votes:
                # Last authority vote wins (in case of duplicates)
                authority_vote = authority_votes[-1]
                winning_option = authority_vote.option
                agreement_ratio = counts[winning_option] / total
                return ConsensusResult(
                    winning_option=winning_option,
                    vote_counts=dict(counts),
                    total_votes=total,
                    agreement_ratio=agreement_ratio,
                    reached_consensus=True,
                    protocol_name=self.name,
                    decided_at=datetime.now(timezone.utc),
                    metadata={"authority_overrode": True, "authority_id": self._authority_id},
                )

            # Fallback to majority vote
            fallback_result = self._fallback.resolve(votes)
            # Return under this protocol's name
            return ConsensusResult(
                winning_option=fallback_result.winning_option,
                vote_counts=fallback_result.vote_counts,
                total_votes=fallback_result.total_votes,
                agreement_ratio=fallback_result.agreement_ratio,
                reached_consensus=fallback_result.reached_consensus,
                protocol_name=self.name,
                decided_at=fallback_result.decided_at,
                metadata={"authority_overrode": False, "authority_id": self._authority_id},
            )
