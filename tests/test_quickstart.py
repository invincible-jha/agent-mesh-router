"""Test that the 3-line quickstart API works for agent-mesh-router."""
from __future__ import annotations


def test_quickstart_import() -> None:
    from agent_mesh_router import Router

    router = Router()
    assert router is not None


def test_quickstart_register_and_route() -> None:
    from agent_mesh_router import Router

    router = Router()
    router.register("worker-1", capabilities=["nlp"])
    result = router.route({"type": "nlp", "content": "Hello"})
    assert result is not None
    assert "agent_id" in result
    assert result["agent_id"] == "worker-1"


def test_quickstart_no_agents_returns_error() -> None:
    from agent_mesh_router import Router

    router = Router()
    result = router.route({"type": "task", "content": "test"})
    assert "error" in result
    assert result["error"] == "no_agents_available"


def test_quickstart_multiple_agents() -> None:
    from agent_mesh_router import Router

    router = Router()
    router.register("worker-1")
    router.register("worker-2")
    result = router.route({"type": "task"})
    assert "agent_id" in result
    assert result["agent_id"] in ("worker-1", "worker-2")


def test_quickstart_table_accessible() -> None:
    from agent_mesh_router import Router
    from agent_mesh_router.routing.table import RoutingTable

    router = Router()
    assert isinstance(router.table, RoutingTable)


def test_quickstart_repr() -> None:
    from agent_mesh_router import Router

    router = Router()
    router.register("agent-1")
    text = repr(router)
    assert "Router" in text
    assert "1" in text
