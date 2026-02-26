"""HealthMonitor — detect and mark unhealthy agents based on heartbeat gaps.

The HealthMonitor wraps a ``FleetRegistry`` and periodically (or on-demand)
checks each registered node's ``last_heartbeat`` timestamp.  If a node has
not sent a heartbeat within the configured ``heartbeat_timeout_seconds``,
the monitor marks it as ``AgentStatus.UNHEALTHY``.

Nodes that recover (send a heartbeat after being marked unhealthy) are
automatically restored to ``AgentStatus.HEALTHY`` on the next heartbeat
call through the registry.

Usage patterns
--------------
* **Passive check**: call ``check_all()`` from an async scheduler or
  immediately before making routing decisions.
* **Background loop**: call ``start()`` to run an asyncio background task
  that calls ``check_all()`` on a fixed interval.

Example
-------
::

    import asyncio
    from agent_mesh_router.fleet.registry import FleetRegistry, AgentNode
    from agent_mesh_router.fleet.health import HealthMonitor

    registry = FleetRegistry()
    registry.register(AgentNode(agent_id="worker-1", capabilities={"task"}))

    monitor = HealthMonitor(registry, heartbeat_timeout_seconds=5.0)
    await monitor.start()

    # After 5+ seconds without a heartbeat from worker-1,
    # it will be marked UNHEALTHY.
    await asyncio.sleep(6)
    node = registry.get("worker-1")
    print(node.status)  # AgentStatus.UNHEALTHY

    await monitor.stop()
"""
from __future__ import annotations

import asyncio
import logging
import time

from agent_mesh_router.fleet.registry import AgentNode, AgentStatus, FleetRegistry

logger = logging.getLogger(__name__)


class HealthMonitor:
    """Monitor agent heartbeat timestamps and mark stale agents UNHEALTHY.

    Parameters
    ----------
    registry:
        The ``FleetRegistry`` to monitor.
    heartbeat_timeout_seconds:
        Number of seconds without a heartbeat before an agent is marked
        UNHEALTHY.  Defaults to 30 seconds.
    check_interval_seconds:
        How often the background loop (started via ``start()``) calls
        ``check_all()``.  Defaults to 10 seconds.
    restore_on_heartbeat:
        When True (default), receiving a heartbeat for an UNHEALTHY agent
        through ``record_heartbeat()`` restores it to HEALTHY.
    """

    def __init__(
        self,
        registry: FleetRegistry,
        *,
        heartbeat_timeout_seconds: float = 30.0,
        check_interval_seconds: float = 10.0,
        restore_on_heartbeat: bool = True,
    ) -> None:
        if heartbeat_timeout_seconds <= 0:
            raise ValueError(
                f"heartbeat_timeout_seconds must be positive, "
                f"got {heartbeat_timeout_seconds}."
            )
        if check_interval_seconds <= 0:
            raise ValueError(
                f"check_interval_seconds must be positive, "
                f"got {check_interval_seconds}."
            )
        self._registry = registry
        self._heartbeat_timeout = heartbeat_timeout_seconds
        self._check_interval = check_interval_seconds
        self._restore_on_heartbeat = restore_on_heartbeat

        self._running: bool = False
        self._monitor_task: asyncio.Task[None] | None = None
        self._total_checks: int = 0
        self._total_marked_unhealthy: int = 0
        self._total_restored: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background health-check loop.

        Safe to call multiple times — subsequent calls are no-ops.
        """
        if self._running:
            return
        self._running = True
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(), name="fleet-health-monitor"
        )
        logger.info(
            "HealthMonitor started (timeout=%.1fs, interval=%.1fs).",
            self._heartbeat_timeout,
            self._check_interval,
        )

    async def stop(self) -> None:
        """Stop the background health-check loop.

        Waits for the current check to complete before returning.
        Safe to call even if ``start()`` was never called.
        """
        if not self._running:
            return
        self._running = False
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        logger.info("HealthMonitor stopped.")

    # ------------------------------------------------------------------
    # Heartbeat recording
    # ------------------------------------------------------------------

    def record_heartbeat(
        self,
        agent_id: str,
        *,
        load_score: float | None = None,
    ) -> None:
        """Record a live heartbeat and optionally restore UNHEALTHY agents.

        Parameters
        ----------
        agent_id:
            The agent sending the heartbeat.
        load_score:
            Optional updated load value (forwarded to the registry).

        Notes
        -----
        If the agent was previously UNHEALTHY and ``restore_on_heartbeat``
        is True, the agent is promoted back to HEALTHY.
        """
        try:
            node = self._registry.get(agent_id)
        except KeyError:
            logger.debug(
                "HealthMonitor: heartbeat from unknown agent %r; ignoring.",
                agent_id,
            )
            return

        restore_to_healthy = (
            self._restore_on_heartbeat
            and node.status == AgentStatus.UNHEALTHY
        )

        new_status = AgentStatus.HEALTHY if restore_to_healthy else None
        self._registry.heartbeat(
            agent_id,
            load_score=load_score,
            status=new_status,
        )

        if restore_to_healthy:
            self._total_restored += 1
            logger.info(
                "HealthMonitor: agent %r restored to HEALTHY after heartbeat.",
                agent_id,
            )

    # ------------------------------------------------------------------
    # Active checking
    # ------------------------------------------------------------------

    def check_all(self) -> list[str]:
        """Check all registered nodes and mark stale ones UNHEALTHY.

        Returns
        -------
        list[str]
            Agent IDs that were newly marked UNHEALTHY during this check.
        """
        now = time.time()
        newly_unhealthy: list[str] = []

        for node in self._registry.list_nodes(healthy_only=False):
            age = now - node.last_heartbeat
            if (
                age > self._heartbeat_timeout
                and node.status != AgentStatus.UNHEALTHY
            ):
                self._mark_unhealthy(node)
                newly_unhealthy.append(node.agent_id)

        self._total_checks += 1
        return newly_unhealthy

    def check_node(self, agent_id: str) -> bool:
        """Check a single node; mark UNHEALTHY if stale.

        Parameters
        ----------
        agent_id:
            The agent to check.

        Returns
        -------
        bool
            True if the node was newly marked UNHEALTHY; False otherwise.

        Raises
        ------
        KeyError
            If the agent is not registered.
        """
        node = self._registry.get(agent_id)
        age = time.time() - node.last_heartbeat

        if age > self._heartbeat_timeout and node.status != AgentStatus.UNHEALTHY:
            self._mark_unhealthy(node)
            return True
        return False

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """Return True if the background monitor loop is active."""
        return self._running

    @property
    def total_checks(self) -> int:
        """Total number of ``check_all()`` calls made."""
        return self._total_checks

    @property
    def total_marked_unhealthy(self) -> int:
        """Total agents marked UNHEALTHY over the monitor's lifetime."""
        return self._total_marked_unhealthy

    @property
    def total_restored(self) -> int:
        """Total agents restored to HEALTHY via ``record_heartbeat()``."""
        return self._total_restored

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _mark_unhealthy(self, node: AgentNode) -> None:
        """Update a node's status to UNHEALTHY."""
        node.status = AgentStatus.UNHEALTHY
        self._total_marked_unhealthy += 1
        logger.warning(
            "HealthMonitor: agent %r marked UNHEALTHY "
            "(last heartbeat %.1f s ago).",
            node.agent_id,
            time.time() - node.last_heartbeat,
        )

    async def _monitor_loop(self) -> None:
        """Background asyncio task: repeatedly call ``check_all()``."""
        while self._running:
            try:
                newly_unhealthy = self.check_all()
                if newly_unhealthy:
                    logger.warning(
                        "HealthMonitor: %d agent(s) newly marked UNHEALTHY: %r",
                        len(newly_unhealthy),
                        newly_unhealthy,
                    )
                await asyncio.sleep(self._check_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("HealthMonitor: unexpected error in monitor loop.")
