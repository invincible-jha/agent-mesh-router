"""A2A Discovery Endpoint — serves Agent Cards at /.well-known/agent.json.

The ``DiscoveryEndpoint`` is a minimal, framework-agnostic HTTP handler that
returns the agent's ``AgentCard`` at the well-known path specified by the
A2A protocol.  It can be embedded into any HTTP server by calling
``handle_request`` and using the returned tuple to construct a framework
response.

Example
-------
::

    from agent_mesh_router.a2a.discovery import DiscoveryEndpoint
    from agent_mesh_router.a2a.models import AgentCard

    card = AgentCard(
        name="MyAgent",
        description="Does useful things.",
        url="https://agents.example.com/my-agent",
    )
    endpoint = DiscoveryEndpoint(card)

    # In a stdlib http.server handler:
    status, headers, body = endpoint.handle_request(self.path)
"""
from __future__ import annotations

from agent_mesh_router.a2a.models import AgentCard

# The canonical well-known path defined by the A2A protocol.
_AGENT_CARD_PATH: str = "/.well-known/agent.json"

# HTTP status codes used by this endpoint.
_HTTP_OK: int = 200
_HTTP_NOT_FOUND: int = 404

# Content-Type header value for JSON responses.
_JSON_CONTENT_TYPE: str = "application/json"


class DiscoveryEndpoint:
    """Serves Agent Cards at ``/.well-known/agent.json``.

    Parameters
    ----------
    card:
        The ``AgentCard`` to serve.  The card is serialized to JSON once
        on first request and cached for subsequent calls.

    Example
    -------
    ::

        endpoint = DiscoveryEndpoint(card)
        status, headers, body = endpoint.handle_request("/.well-known/agent.json")
        # status == 200
        # headers == {"Content-Type": "application/json"}
        # body contains the JSON-serialized agent card
    """

    def __init__(self, card: AgentCard) -> None:
        self._card = card
        self._cached_json: str | None = None

    @property
    def card(self) -> AgentCard:
        """The Agent Card being served by this endpoint."""
        return self._card

    def handle_request(
        self, path: str
    ) -> tuple[int, dict[str, str], str]:
        """Handle an incoming HTTP request for the agent card.

        Only requests to the exact path ``/.well-known/agent.json`` return
        a 200 with the card body.  All other paths return 404 with an empty
        body.

        Parameters
        ----------
        path:
            The HTTP request path (e.g. ``"/.well-known/agent.json"`` or
            ``"/health"``).

        Returns
        -------
        tuple[int, dict[str, str], str]
            A three-tuple of ``(status_code, headers_dict, body_string)``.
            For a 200 response the headers include ``Content-Type: application/json``.
            For a 404 response the headers dict and body are empty.
        """
        normalized_path = path.split("?")[0]  # strip query string

        if normalized_path == _AGENT_CARD_PATH:
            body = self._get_or_build_json()
            headers: dict[str, str] = {"Content-Type": _JSON_CONTENT_TYPE}
            return _HTTP_OK, headers, body

        return _HTTP_NOT_FOUND, {}, ""

    def invalidate_cache(self) -> None:
        """Clear the cached JSON so the next request rebuilds from the card.

        Call this after mutating ``card`` fields directly (not recommended
        in production; prefer constructing a new ``DiscoveryEndpoint``).
        """
        self._cached_json = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_build_json(self) -> str:
        """Return the cached JSON serialization of the card, building it if needed."""
        if self._cached_json is None:
            self._cached_json = self._card.model_dump_json(indent=2)
        return self._cached_json
