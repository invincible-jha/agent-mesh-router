"""CLI entry point for agent-mesh-router.

Invoked as::

    agent-mesh-router [OPTIONS] COMMAND [ARGS]...

or, during development::

    python -m agent_mesh_router.cli.main
"""
from __future__ import annotations

import click
from rich.console import Console

from agent_mesh_router.cli.resilience_commands import resilience_group

console = Console()


@click.group()
@click.version_option()
def cli() -> None:
    """Multi-agent communication, task routing, and workflow orchestration"""


@cli.command(name="version")
def version_command() -> None:
    """Show detailed version information."""
    from agent_mesh_router import __version__

    console.print(f"[bold]agent-mesh-router[/bold] v{__version__}")


@cli.command(name="plugins")
def plugins_command() -> None:
    """List all registered plugins loaded from entry-points."""
    console.print("[bold]Registered plugins:[/bold]")
    console.print("  (No plugins registered. Install a plugin package to see entries here.)")


cli.add_command(resilience_group)


if __name__ == "__main__":
    cli()
