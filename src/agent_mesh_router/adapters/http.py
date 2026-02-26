"""HttpTransport — deliver MessageEnvelopes via HTTP POST.

Envelopes are serialized to JSON and sent as the request body with
``Content-Type: application/json``.  The transport supports configurable
retry logic with exponential backoff.

Optional dependency
-------------------
Requires ``httpx`` (async HTTP client).  Import is guarded so the rest of
the package remains importable even when ``httpx`` is not installed.

Install with::

    pip install agent-mesh-router[http]

Example
-------
::

    import asyncio
    from agent_mesh_router.adapters.http import HttpTransport
    from agent_mesh_router.messages.envelope import MessageEnvelope

    transport = HttpTransport(
        base_url="https://agent-gateway.example.com",
        max_retries=3,
        timeout_seconds=5.0,
    )

    envelope = MessageEnvelope(
        sender="agent-a", receiver="agent-b", payload={"task": "run"}
    )
    asyncio.run(transport.send(envelope))
"""
from __future__ import annotations

import asyncio
import logging

from agent_mesh_router.messages.envelope import MessageEnvelope

logger = logging.getLogger(__name__)

try:
    import httpx

    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False


class HttpTransportError(RuntimeError):
    """Raised when an envelope cannot be delivered after all retry attempts."""


class HttpTransport:
    """Send ``MessageEnvelope`` objects via HTTP POST with retry/backoff.

    Parameters
    ----------
    base_url:
        Base URL of the agent gateway or receiving endpoint.
        Individual envelopes are posted to ``{base_url}/messages``.
    max_retries:
        Number of additional attempts after the initial failure.
        Total attempts = ``max_retries + 1``.
    timeout_seconds:
        Per-request timeout in seconds.
    backoff_base_seconds:
        Initial backoff duration.  Each retry waits
        ``backoff_base_seconds * 2^(attempt-1)`` seconds (capped at 30 s).
    headers:
        Extra HTTP headers merged into every request.

    Raises
    ------
    ImportError
        At construction time if ``httpx`` is not installed.
    """

    def __init__(
        self,
        base_url: str,
        *,
        max_retries: int = 3,
        timeout_seconds: float = 10.0,
        backoff_base_seconds: float = 0.5,
        headers: dict[str, str] | None = None,
    ) -> None:
        if not _HTTPX_AVAILABLE:
            raise ImportError(
                "HttpTransport requires 'httpx'. "
                "Install it with: pip install agent-mesh-router[http]"
            )
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}.")
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}.")

        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._timeout_seconds = timeout_seconds
        self._backoff_base = backoff_base_seconds
        self._extra_headers = headers or {}
        self._total_sent: int = 0
        self._total_failed: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def send(self, envelope: MessageEnvelope) -> dict[str, object]:
        """Send an envelope via HTTP POST.

        Retries on transient failures (network errors, 5xx responses)
        using exponential backoff.

        Parameters
        ----------
        envelope:
            The envelope to transmit.

        Returns
        -------
        dict[str, object]
            Parsed JSON response body from the receiver.

        Raises
        ------
        HttpTransportError
            If all retry attempts are exhausted without a successful response.
        """
        url = f"{self._base_url}/messages"
        payload_bytes = envelope.to_json()
        headers = {
            "Content-Type": "application/json",
            "X-Trace-Id": envelope.trace_id,
            "X-Message-Id": envelope.message_id,
            **self._extra_headers,
        }

        last_exception: Exception | None = None

        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                backoff = min(self._backoff_base * (2 ** (attempt - 1)), 30.0)
                logger.debug(
                    "HttpTransport: retry %d/%d for message %s in %.2f s.",
                    attempt,
                    self._max_retries,
                    envelope.message_id[:8],
                    backoff,
                )
                await asyncio.sleep(backoff)

            try:
                async with httpx.AsyncClient(
                    timeout=self._timeout_seconds
                ) as client:
                    response = await client.post(
                        url,
                        content=payload_bytes,
                        headers=headers,
                    )

                if response.is_success:
                    self._total_sent += 1
                    logger.debug(
                        "HttpTransport: delivered message %s (status %d).",
                        envelope.message_id[:8],
                        response.status_code,
                    )
                    try:
                        return response.json()
                    except Exception:
                        return {"status_code": response.status_code}

                # Treat 4xx as permanent failures (no retry)
                if 400 <= response.status_code < 500:
                    self._total_failed += 1
                    raise HttpTransportError(
                        f"Permanent HTTP {response.status_code} error "
                        f"for message {envelope.message_id[:8]}."
                    )

                # 5xx — transient; will retry
                last_exception = HttpTransportError(
                    f"HTTP {response.status_code} error "
                    f"(attempt {attempt + 1}/{self._max_retries + 1})."
                )

            except HttpTransportError:
                raise
            except Exception as exc:
                last_exception = exc
                logger.warning(
                    "HttpTransport: attempt %d failed for message %s: %s",
                    attempt + 1,
                    envelope.message_id[:8],
                    exc,
                )

        self._total_failed += 1
        raise HttpTransportError(
            f"Failed to deliver message {envelope.message_id[:8]} "
            f"after {self._max_retries + 1} attempt(s)."
        ) from last_exception

    async def send_many(
        self, envelopes: list[MessageEnvelope]
    ) -> list[dict[str, object]]:
        """Send multiple envelopes concurrently.

        Parameters
        ----------
        envelopes:
            List of envelopes to send.

        Returns
        -------
        list[dict[str, object]]
            Response dicts corresponding to each envelope (in order).
            Failures are represented as ``{"error": "..."}`` dicts.
        """
        import asyncio as _asyncio

        async def _safe_send(env: MessageEnvelope) -> dict[str, object]:
            try:
                return await self.send(env)
            except Exception as exc:
                return {"error": str(exc), "message_id": env.message_id}

        return list(
            await _asyncio.gather(*[_safe_send(e) for e in envelopes])
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def base_url(self) -> str:
        """Configured base URL."""
        return self._base_url

    @property
    def total_sent(self) -> int:
        """Total successfully delivered envelopes."""
        return self._total_sent

    @property
    def total_failed(self) -> int:
        """Total permanently failed delivery attempts."""
        return self._total_failed

    def __repr__(self) -> str:
        return (
            f"HttpTransport("
            f"base_url={self._base_url!r}, "
            f"max_retries={self._max_retries}, "
            f"sent={self._total_sent}, "
            f"failed={self._total_failed}"
            f")"
        )
