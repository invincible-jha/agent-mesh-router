"""Agent consensus protocol — multi-agent agreement mechanisms.

Provides ABC and three commodity implementations:

MajorityVote
    Simple majority (> 50% agreement) for binary or multi-choice decisions.
ConfidenceWeighted
    Each voter supplies a confidence score; weighted majority wins.
Hierarchical
    A designated authority agent's vote overrides peers.

Classes
-------
ConsensusProtocol
    Abstract base class all implementations must satisfy.
VoteRecord
    An individual agent vote with confidence and metadata.
ConsensusResult
    Immutable outcome of a consensus round.
MajorityVote
    Plurality / majority consensus.
ConfidenceWeighted
    Confidence-weighted consensus.
Hierarchical
    Authority-first consensus with fallback to majority.
"""
from __future__ import annotations

from agent_mesh_router.consensus.protocol import (
    ConsensusProtocol,
    ConsensusResult,
    VoteRecord,
    MajorityVote,
    ConfidenceWeighted,
    Hierarchical,
)

__all__ = [
    "ConfidenceWeighted",
    "ConsensusProtocol",
    "ConsensusResult",
    "Hierarchical",
    "MajorityVote",
    "VoteRecord",
]
