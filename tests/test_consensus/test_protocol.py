"""Tests for agent_mesh_router.consensus.protocol."""
from __future__ import annotations

import threading

import pytest

from agent_mesh_router.consensus.protocol import (
    ConfidenceWeighted,
    ConsensusProtocol,
    ConsensusResult,
    Hierarchical,
    MajorityVote,
    VoteRecord,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vote(agent_id: str, option: str, confidence: float = 1.0) -> VoteRecord:
    return VoteRecord(agent_id=agent_id, option=option, confidence=confidence)


# ===========================================================================
# VoteRecord
# ===========================================================================


class TestVoteRecord:
    def test_frozen(self) -> None:
        vote = _vote("a", "yes")
        with pytest.raises(Exception):
            vote.option = "no"  # type: ignore[misc]

    def test_empty_agent_id_raises(self) -> None:
        with pytest.raises(ValueError, match="agent_id"):
            VoteRecord(agent_id="", option="yes")

    def test_empty_option_raises(self) -> None:
        with pytest.raises(ValueError, match="option"):
            VoteRecord(agent_id="a", option="")

    def test_confidence_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            VoteRecord(agent_id="a", option="yes", confidence=1.5)

    def test_negative_confidence_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            VoteRecord(agent_id="a", option="yes", confidence=-0.1)

    def test_default_confidence(self) -> None:
        vote = _vote("a", "yes")
        assert vote.confidence == 1.0

    def test_metadata_defaults_to_empty(self) -> None:
        vote = _vote("a", "yes")
        assert vote.metadata == {}


# ===========================================================================
# ConsensusResult
# ===========================================================================


class TestConsensusResult:
    def _make_result(self, option: str = "yes", total: int = 3) -> ConsensusResult:
        from datetime import datetime, timezone

        return ConsensusResult(
            winning_option=option,
            vote_counts={option: 2, "no": 1},
            total_votes=total,
            agreement_ratio=2 / total,
            reached_consensus=True,
            protocol_name="test",
            decided_at=datetime.now(timezone.utc),
        )

    def test_frozen(self) -> None:
        result = self._make_result()
        with pytest.raises(Exception):
            result.winning_option = "no"  # type: ignore[misc]

    def test_summary_with_consensus(self) -> None:
        result = self._make_result("deploy")
        summary = result.summary()
        assert "deploy" in summary
        assert "test" in summary

    def test_summary_no_consensus(self) -> None:
        from datetime import datetime, timezone

        result = ConsensusResult(
            winning_option="",
            vote_counts={},
            total_votes=0,
            agreement_ratio=0.0,
            reached_consensus=False,
            protocol_name="test",
            decided_at=datetime.now(timezone.utc),
        )
        assert "No consensus" in result.summary()

    def test_to_dict_keys(self) -> None:
        result = self._make_result()
        d = result.to_dict()
        for key in ("winning_option", "vote_counts", "total_votes", "agreement_ratio",
                    "reached_consensus", "protocol_name", "decided_at"):
            assert key in d


# ===========================================================================
# MajorityVote
# ===========================================================================


class TestMajorityVote:
    def test_is_consensus_protocol(self) -> None:
        assert isinstance(MajorityVote(), ConsensusProtocol)

    def test_name(self) -> None:
        assert MajorityVote().name == "majority_vote"

    def test_empty_votes_no_consensus(self) -> None:
        result = MajorityVote().resolve([])
        assert not result.reached_consensus

    def test_single_vote_wins(self) -> None:
        result = MajorityVote().resolve([_vote("a", "yes")])
        assert result.winning_option == "yes"
        assert result.reached_consensus

    def test_majority_wins(self) -> None:
        votes = [_vote("a", "yes"), _vote("b", "yes"), _vote("c", "no")]
        result = MajorityVote().resolve(votes)
        assert result.winning_option == "yes"
        assert result.reached_consensus

    def test_tiebreak_lexicographic(self) -> None:
        votes = [_vote("a", "beta"), _vote("b", "alpha")]
        protocol = MajorityVote(require_majority=False)
        result = protocol.resolve(votes)
        assert result.winning_option == "alpha"

    def test_no_majority_when_tied_with_require_majority(self) -> None:
        votes = [_vote("a", "yes"), _vote("b", "no")]
        result = MajorityVote(require_majority=True).resolve(votes)
        assert not result.reached_consensus

    def test_no_majority_false_accepts_plurality(self) -> None:
        votes = [_vote("a", "yes"), _vote("b", "no")]
        result = MajorityVote(require_majority=False).resolve(votes)
        assert result.reached_consensus

    def test_vote_counts_correct(self) -> None:
        votes = [_vote("a", "yes"), _vote("b", "yes"), _vote("c", "no")]
        result = MajorityVote().resolve(votes)
        assert result.vote_counts == {"yes": 2, "no": 1}
        assert result.total_votes == 3

    def test_agreement_ratio(self) -> None:
        votes = [_vote("a", "x"), _vote("b", "x"), _vote("c", "x")]
        result = MajorityVote().resolve(votes)
        assert abs(result.agreement_ratio - 1.0) < 1e-9

    def test_protocol_name_in_result(self) -> None:
        result = MajorityVote().resolve([_vote("a", "yes")])
        assert result.protocol_name == "majority_vote"


# ===========================================================================
# ConfidenceWeighted
# ===========================================================================


class TestConfidenceWeighted:
    def test_is_consensus_protocol(self) -> None:
        assert isinstance(ConfidenceWeighted(), ConsensusProtocol)

    def test_name(self) -> None:
        assert ConfidenceWeighted().name == "confidence_weighted"

    def test_empty_votes_no_consensus(self) -> None:
        result = ConfidenceWeighted().resolve([])
        assert not result.reached_consensus

    def test_high_confidence_wins(self) -> None:
        votes = [
            _vote("a", "yes", confidence=0.9),
            _vote("b", "no", confidence=0.1),
        ]
        result = ConfidenceWeighted().resolve(votes)
        assert result.winning_option == "yes"

    def test_low_confidence_loses(self) -> None:
        votes = [
            _vote("a", "yes", confidence=0.2),
            _vote("b", "no", confidence=0.8),
        ]
        result = ConfidenceWeighted().resolve(votes)
        assert result.winning_option == "no"

    def test_zero_confidence_falls_back_to_count(self) -> None:
        votes = [
            _vote("a", "yes", confidence=0.0),
            _vote("b", "yes", confidence=0.0),
            _vote("c", "no", confidence=0.0),
        ]
        result = ConfidenceWeighted().resolve(votes)
        assert result.winning_option == "yes"

    def test_min_winning_weight_ratio(self) -> None:
        protocol = ConfidenceWeighted(min_winning_weight_ratio=0.9)
        votes = [
            _vote("a", "yes", confidence=0.6),
            _vote("b", "no", confidence=0.4),
        ]
        result = protocol.resolve(votes)
        # 0.6 weight vs total 1.0 = 60% < 90%
        assert not result.reached_consensus

    def test_invalid_min_ratio_raises(self) -> None:
        with pytest.raises(ValueError, match="min_winning_weight_ratio"):
            ConfidenceWeighted(min_winning_weight_ratio=1.1)

    def test_metadata_includes_weights(self) -> None:
        votes = [_vote("a", "yes", confidence=0.7)]
        result = ConfidenceWeighted().resolve(votes)
        assert "total_weight" in result.metadata
        assert "winning_weight" in result.metadata


# ===========================================================================
# Hierarchical
# ===========================================================================


class TestHierarchical:
    def test_is_consensus_protocol(self) -> None:
        assert isinstance(Hierarchical("lead"), ConsensusProtocol)

    def test_name(self) -> None:
        assert Hierarchical("lead").name == "hierarchical"

    def test_empty_authority_id_raises(self) -> None:
        with pytest.raises(ValueError, match="authority_agent_id"):
            Hierarchical("")

    def test_empty_votes_no_consensus(self) -> None:
        result = Hierarchical("lead").resolve([])
        assert not result.reached_consensus

    def test_authority_vote_overrides(self) -> None:
        protocol = Hierarchical(authority_agent_id="lead")
        votes = [
            _vote("lead", "deploy"),
            _vote("worker_a", "rollback"),
            _vote("worker_b", "rollback"),
        ]
        result = protocol.resolve(votes)
        assert result.winning_option == "deploy"
        assert result.reached_consensus
        assert result.metadata.get("authority_overrode") is True

    def test_fallback_to_majority_without_authority(self) -> None:
        protocol = Hierarchical(authority_agent_id="lead")
        votes = [
            _vote("worker_a", "yes"),
            _vote("worker_b", "yes"),
            _vote("worker_c", "no"),
        ]
        result = protocol.resolve(votes)
        assert result.winning_option == "yes"
        assert result.metadata.get("authority_overrode") is False

    def test_authority_property(self) -> None:
        protocol = Hierarchical(authority_agent_id="boss")
        assert protocol.authority_agent_id == "boss"

    def test_protocol_name_in_result(self) -> None:
        votes = [_vote("a", "yes")]
        result = Hierarchical("a").resolve(votes)
        assert result.protocol_name == "hierarchical"

    def test_last_authority_vote_wins(self) -> None:
        protocol = Hierarchical(authority_agent_id="lead")
        votes = [
            _vote("lead", "rollback"),
            _vote("lead", "deploy"),  # second vote from authority
        ]
        result = protocol.resolve(votes)
        assert result.winning_option == "deploy"


# ===========================================================================
# Thread safety
# ===========================================================================


class TestThreadSafety:
    def test_concurrent_majority_vote(self) -> None:
        protocol = MajorityVote()
        results: list[ConsensusResult] = []
        errors: list[Exception] = []

        def run() -> None:
            try:
                votes = [_vote(f"agent_{i}", "yes") for i in range(5)]
                result = protocol.resolve(votes)
                results.append(result)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=run) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert all(r.winning_option == "yes" for r in results)
