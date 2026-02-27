"""Tests for agent_mesh_router.quotas.resource_quota."""
from __future__ import annotations

import threading

import pytest

from agent_mesh_router.quotas.resource_quota import (
    QuotaEnforcer,
    QuotaUsage,
    QuotaViolation,
    ResourceQuota,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quota(
    agent_id: str = "agent_a",
    tokens: float | None = 1000.0,
    api_calls: float | None = 100.0,
) -> ResourceQuota:
    return ResourceQuota(
        agent_id=agent_id,
        token_limit=tokens,
        api_call_limit=api_calls,
    )


def _enforcer_with_quota(**kwargs: object) -> tuple[QuotaEnforcer, ResourceQuota]:
    q = _quota(**kwargs)  # type: ignore[arg-type]
    e = QuotaEnforcer()
    e.register(q)
    return e, q


# ===========================================================================
# ResourceQuota
# ===========================================================================


class TestResourceQuota:
    def test_frozen(self) -> None:
        q = _quota()
        with pytest.raises(Exception):
            q.agent_id = "other"  # type: ignore[misc]

    def test_empty_agent_id_raises(self) -> None:
        with pytest.raises(ValueError, match="agent_id"):
            ResourceQuota(agent_id="")

    def test_negative_token_limit_raises(self) -> None:
        with pytest.raises(ValueError, match="token_limit"):
            ResourceQuota(agent_id="a", token_limit=-1.0)

    def test_none_limits_allowed(self) -> None:
        q = ResourceQuota(agent_id="a")
        assert q.token_limit is None
        assert q.api_call_limit is None

    def test_dimensions_returns_all_limits(self) -> None:
        q = ResourceQuota(agent_id="a", token_limit=500.0, api_call_limit=50.0)
        dims = q.dimensions
        assert "tokens" in dims
        assert "api_calls" in dims
        assert dims["tokens"] == 500.0

    def test_default_label_empty(self) -> None:
        q = ResourceQuota(agent_id="a")
        assert q.label == ""


# ===========================================================================
# QuotaViolation
# ===========================================================================


class TestQuotaViolation:
    def test_is_runtime_error(self) -> None:
        exc = QuotaViolation("a", "tokens", 100.0, 50.0, 500.0)
        assert isinstance(exc, RuntimeError)

    def test_message_contains_key_info(self) -> None:
        exc = QuotaViolation("agent_x", "tokens", 200.0, 10.0, 500.0)
        msg = str(exc)
        assert "agent_x" in msg
        assert "tokens" in msg

    def test_attributes(self) -> None:
        exc = QuotaViolation("a", "api_calls", 5.0, 3.0, 10.0)
        assert exc.agent_id == "a"
        assert exc.dimension == "api_calls"
        assert exc.requested == 5.0
        assert exc.remaining == 3.0
        assert exc.limit == 10.0


# ===========================================================================
# QuotaUsage
# ===========================================================================


class TestQuotaUsage:
    def test_frozen(self) -> None:
        from datetime import datetime, timezone

        usage = QuotaUsage(
            agent_id="a",
            tokens_used=0.0,
            memory_mb_used=0.0,
            cpu_seconds_used=0.0,
            api_calls_used=0.0,
            snapshotted_at=datetime.now(timezone.utc),
        )
        with pytest.raises(Exception):
            usage.tokens_used = 99.0  # type: ignore[misc]

    def test_to_dict_keys(self) -> None:
        from datetime import datetime, timezone

        usage = QuotaUsage(
            agent_id="a",
            tokens_used=10.0,
            memory_mb_used=0.0,
            cpu_seconds_used=0.0,
            api_calls_used=0.0,
            snapshotted_at=datetime.now(timezone.utc),
        )
        d = usage.to_dict()
        assert "agent_id" in d
        assert "tokens_used" in d
        assert "snapshotted_at" in d


# ===========================================================================
# QuotaEnforcer — construction
# ===========================================================================


class TestQuotaEnforcerConstruction:
    def test_empty_on_init(self) -> None:
        e = QuotaEnforcer()
        assert len(e) == 0
        assert e.registered_agents() == []

    def test_register_adds_agent(self) -> None:
        e, _ = _enforcer_with_quota()
        assert "agent_a" in e
        assert len(e) == 1

    def test_register_multiple(self) -> None:
        e = QuotaEnforcer()
        e.register(ResourceQuota("a"))
        e.register(ResourceQuota("b"))
        assert len(e) == 2

    def test_unregister_returns_true(self) -> None:
        e, _ = _enforcer_with_quota()
        assert e.unregister("agent_a") is True
        assert "agent_a" not in e

    def test_unregister_nonexistent_returns_false(self) -> None:
        e = QuotaEnforcer()
        assert e.unregister("missing") is False

    def test_registered_agents_sorted(self) -> None:
        e = QuotaEnforcer()
        e.register(ResourceQuota("c"))
        e.register(ResourceQuota("a"))
        e.register(ResourceQuota("b"))
        assert e.registered_agents() == ["a", "b", "c"]


# ===========================================================================
# QuotaEnforcer — check
# ===========================================================================


class TestCheck:
    def test_check_within_limit_returns_true(self) -> None:
        e, _ = _enforcer_with_quota(tokens=1000.0)
        assert e.check("agent_a", tokens=500.0) is True

    def test_check_exceeds_limit_returns_false(self) -> None:
        e, _ = _enforcer_with_quota(tokens=100.0)
        assert e.check("agent_a", tokens=200.0) is False

    def test_check_raises_on_violation(self) -> None:
        e, _ = _enforcer_with_quota(tokens=100.0)
        with pytest.raises(QuotaViolation) as exc_info:
            e.check("agent_a", tokens=200.0, raise_on_violation=True)
        assert exc_info.value.dimension == "tokens"

    def test_check_none_limit_always_passes(self) -> None:
        e = QuotaEnforcer()
        e.register(ResourceQuota("a"))  # all limits None
        assert e.check("a", tokens=1_000_000.0) is True

    def test_check_unknown_agent_raises_key_error(self) -> None:
        e = QuotaEnforcer()
        with pytest.raises(KeyError):
            e.check("unknown", tokens=10.0)


# ===========================================================================
# QuotaEnforcer — record
# ===========================================================================


class TestRecord:
    def test_record_increases_usage(self) -> None:
        e, _ = _enforcer_with_quota(tokens=1000.0)
        e.record("agent_a", tokens=300.0)
        usage = e.usage("agent_a")
        assert usage.tokens_used == 300.0

    def test_record_accumulates(self) -> None:
        e, _ = _enforcer_with_quota(tokens=1000.0)
        e.record("agent_a", tokens=200.0)
        e.record("agent_a", tokens=300.0)
        assert e.usage("agent_a").tokens_used == 500.0

    def test_record_negative_raises(self) -> None:
        e, _ = _enforcer_with_quota(tokens=1000.0)
        with pytest.raises(ValueError, match="non-negative"):
            e.record("agent_a", tokens=-1.0)

    def test_record_unknown_agent_raises(self) -> None:
        e = QuotaEnforcer()
        with pytest.raises(KeyError):
            e.record("unknown", tokens=10.0)

    def test_record_multiple_dimensions(self) -> None:
        e = QuotaEnforcer()
        e.register(ResourceQuota("a", token_limit=1000.0, api_call_limit=50.0))
        e.record("a", tokens=100.0, api_calls=5.0)
        usage = e.usage("a")
        assert usage.tokens_used == 100.0
        assert usage.api_calls_used == 5.0


# ===========================================================================
# QuotaEnforcer — check_and_record
# ===========================================================================


class TestCheckAndRecord:
    def test_check_and_record_records_on_success(self) -> None:
        e, _ = _enforcer_with_quota(tokens=1000.0)
        result = e.check_and_record("agent_a", tokens=200.0)
        assert result is True
        assert e.usage("agent_a").tokens_used == 200.0

    def test_check_and_record_raises_on_violation(self) -> None:
        e, _ = _enforcer_with_quota(tokens=100.0)
        with pytest.raises(QuotaViolation):
            e.check_and_record("agent_a", tokens=500.0, raise_on_violation=True)
        # Usage should NOT be recorded
        assert e.usage("agent_a").tokens_used == 0.0

    def test_check_and_record_returns_false_no_raise(self) -> None:
        e, _ = _enforcer_with_quota(tokens=100.0)
        result = e.check_and_record("agent_a", tokens=500.0, raise_on_violation=False)
        assert result is False
        assert e.usage("agent_a").tokens_used == 0.0


# ===========================================================================
# QuotaEnforcer — reset
# ===========================================================================


class TestReset:
    def test_reset_clears_usage(self) -> None:
        e, _ = _enforcer_with_quota(tokens=1000.0)
        e.record("agent_a", tokens=500.0)
        e.reset("agent_a")
        assert e.usage("agent_a").tokens_used == 0.0

    def test_reset_all_clears_all_agents(self) -> None:
        e = QuotaEnforcer()
        e.register(ResourceQuota("a", token_limit=100.0))
        e.register(ResourceQuota("b", token_limit=100.0))
        e.record("a", tokens=50.0)
        e.record("b", tokens=50.0)
        e.reset_all()
        assert e.usage("a").tokens_used == 0.0
        assert e.usage("b").tokens_used == 0.0

    def test_reset_unknown_raises(self) -> None:
        e = QuotaEnforcer()
        with pytest.raises(KeyError):
            e.reset("unknown")


# ===========================================================================
# QuotaEnforcer — exhaustion
# ===========================================================================


class TestExhaustion:
    def test_check_fails_after_exhaustion(self) -> None:
        e, _ = _enforcer_with_quota(tokens=100.0)
        e.record("agent_a", tokens=100.0)
        assert not e.check("agent_a", tokens=1.0)

    def test_check_after_reset_passes_again(self) -> None:
        e, _ = _enforcer_with_quota(tokens=100.0)
        e.record("agent_a", tokens=100.0)
        e.reset("agent_a")
        assert e.check("agent_a", tokens=50.0)


# ===========================================================================
# Thread safety
# ===========================================================================


class TestThreadSafety:
    def test_concurrent_record(self) -> None:
        e = QuotaEnforcer()
        e.register(ResourceQuota("agent_a", token_limit=1_000_000.0))
        errors: list[Exception] = []

        def worker() -> None:
            try:
                e.record("agent_a", tokens=1.0)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert abs(e.usage("agent_a").tokens_used - 100.0) < 1e-9
