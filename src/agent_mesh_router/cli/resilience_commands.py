"""CLI commands for the resilience layer.

Registered as a sub-group ``resilience`` on the root ``cli`` group.

Usage::

    agent-mesh-router resilience health --endpoints endpoints.json
    agent-mesh-router resilience circuit-breaker status
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group(name="resilience")
def resilience_group() -> None:
    """Service mesh resilience tools: health checks, circuit breakers."""


# ---------------------------------------------------------------------------
# health sub-command
# ---------------------------------------------------------------------------


@resilience_group.command(name="health")
@click.option(
    "--endpoints",
    "endpoints_file",
    required=True,
    type=click.Path(exists=True, readable=True, path_type=Path),
    help="Path to a JSON file containing endpoint definitions.",
)
@click.option(
    "--timeout",
    default=5.0,
    show_default=True,
    type=float,
    help="Probe timeout in seconds.",
)
def health_command(endpoints_file: Path, timeout: float) -> None:
    """Check the health of endpoints defined in a JSON file.

    The JSON file must contain a list of endpoint objects with the fields:
    ``name``, ``host``, ``port`` (and optionally ``weight``, ``healthy``).

    Example JSON::

        [
            {"name": "api-1", "host": "10.0.0.1", "port": 8080},
            {"name": "api-2", "host": "10.0.0.2", "port": 8080}
        ]
    """
    import socket

    from agent_mesh_router.resilience.health_check import (
        HealthCheckConfig,
        HealthChecker,
        HealthStatus,
    )
    from agent_mesh_router.resilience.load_balancer import ServiceEndpoint

    try:
        raw_text = endpoints_file.read_text(encoding="utf-8")
        raw_data: object = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        console.print(f"[bold red]ERROR[/bold red]: Failed to parse JSON: {exc}")
        sys.exit(1)

    if not isinstance(raw_data, list):
        console.print("[bold red]ERROR[/bold red]: JSON file must contain a list of endpoint objects.")
        sys.exit(1)

    endpoints: list[ServiceEndpoint] = []
    for item in raw_data:
        if not isinstance(item, dict):
            console.print(f"[bold red]ERROR[/bold red]: Each endpoint must be a JSON object, got: {item!r}")
            sys.exit(1)
        try:
            endpoints.append(
                ServiceEndpoint(
                    name=item["name"],
                    host=item["host"],
                    port=int(item["port"]),
                    weight=float(item.get("weight", 1.0)),
                    healthy=bool(item.get("healthy", True)),
                )
            )
        except (KeyError, ValueError, TypeError) as exc:
            console.print(f"[bold red]ERROR[/bold red]: Invalid endpoint definition: {exc}")
            sys.exit(1)

    if not endpoints:
        console.print("[yellow]No endpoints found in file.[/yellow]")
        return

    config = HealthCheckConfig(
        interval_seconds=60.0,
        timeout_seconds=timeout,
        healthy_threshold=1,
        unhealthy_threshold=1,
    )
    checker = HealthChecker(config)

    def tcp_probe(endpoint: ServiceEndpoint) -> None:
        """Attempt a TCP connection to verify the endpoint is reachable."""
        with socket.create_connection(
            (endpoint.host, endpoint.port), timeout=timeout
        ):
            pass

    console.print(f"\n[bold]Checking {len(endpoints)} endpoint(s)...[/bold]\n")
    results = checker.check_all(endpoints, tcp_probe)

    table = Table(title="Endpoint Health")
    table.add_column("Name", style="cyan")
    table.add_column("Host")
    table.add_column("Port")
    table.add_column("Status")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Details")

    exit_code = 0
    for result in results:
        status_str = result.status.value.upper()
        if result.status == HealthStatus.HEALTHY:
            status_display = f"[green]{status_str}[/green]"
        elif result.status == HealthStatus.DEGRADED:
            status_display = f"[yellow]{status_str}[/yellow]"
            exit_code = 2
        else:
            status_display = f"[red]{status_str}[/red]"
            exit_code = 1

        # Find matching endpoint for host/port display
        matching = next((e for e in endpoints if e.name == result.endpoint_name), None)
        host = matching.host if matching else "?"
        port = str(matching.port) if matching else "?"

        details_str = (
            result.details.get("error_message", "") if result.details else ""
        )
        table.add_row(
            result.endpoint_name,
            host,
            port,
            status_display,
            f"{result.latency_ms:.1f}",
            details_str[:60] + ("..." if len(details_str) > 60 else ""),
        )

    console.print(table)
    sys.exit(exit_code)


# ---------------------------------------------------------------------------
# circuit-breaker sub-command
# ---------------------------------------------------------------------------


@resilience_group.command(name="circuit-breaker")
@click.argument("subcommand", type=click.Choice(["status"]))
def circuit_breaker_command(subcommand: str) -> None:
    """Inspect circuit breaker state.

    SUBCOMMAND must be 'status'.

    Note: circuit breaker instances are created per-process and are not
    persisted between CLI invocations.  Use this command from within a
    running service process or extend it with a shared state backend.
    """
    if subcommand == "status":
        console.print(
            "[bold]Circuit breaker status[/bold]\n"
            "\nNo circuit breakers are registered in this process.\n"
            "Instantiate a [cyan]CircuitBreaker[/cyan] in your service code and "
            "expose its [cyan].state[/cyan] property via your own status endpoint."
        )
