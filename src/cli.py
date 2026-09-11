"""Command Line Interface (CLI) for Hiver AI Support & Triage Agent."""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import (
    FRUSTRATION_THRESHOLD,
    MIN_INTENT_CONFIDENCE,
    MIN_RETRIEVAL_SIMILARITY,
    TARGET_BRAND,
)
from src.logging_config import setup_logging
from src.models import TriageAction, TweetInput
from src.pipeline import SupportPipeline

app = typer.Typer(help="Hiver AI Support & Triage Agent CLI")
console = Console()

# Diagnostics go to logging; everything the user is meant to read goes to the
# rich console below. Deliberately separate -- routing CLI output through logging
# would mean a user sees nothing unless they also set LOG_LEVEL.
setup_logging()


@app.command()
def process(
    text: str = typer.Option(..., "--text", "-t", help="Raw tweet text to analyze"),
    tweet_id: str = typer.Option("cli_test_001", "--id", help="Optional tweet ID"),
    author_id: str = typer.Option("customer_user", "--author", help="Author ID"),
):
    """Processes a customer tweet and renders intent, triage action, and grounded draft."""
    console.print(f"[bold cyan]Processing tweet for {TARGET_BRAND}...[/bold cyan]\n")

    tweet = TweetInput(tweet_id=tweet_id, text=text, author_id=author_id)
    pipeline = SupportPipeline()
    response = pipeline.process(tweet)

    # Render result table / card
    table = Table(title="[bold green]AI Support Pipeline Decision[/bold green]", show_header=True)
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("Tweet ID", response.tweet_id)
    table.add_row("Input Text", text)
    table.add_row("Classified Intent", f"[bold yellow]{response.intent.primary_intent.value}[/bold yellow] (conf: {response.intent.confidence:.2f})")

    action_color = "green" if response.triage.action == TriageAction.AUTO_HANDLE else "bold red"
    table.add_row("Triage Action", f"[{action_color}]{response.triage.action.value}[/{action_color}]")
    table.add_row("Stated Reason", response.triage.stated_reason)

    if response.drafted_reply:
        table.add_row("Drafted Reply", f"[italic green]\"{response.drafted_reply}\"[/italic green]")
    else:
        table.add_row("Drafted Reply", "[dim italic](Withheld — Ticket Escalated to Human Agent)[/dim italic]")

    table.add_row("Execution Latency", f"{response.execution_time_ms} ms")

    console.print(table)


@app.command()
def info():
    """Displays system configuration and operational thresholds."""
    panel = Panel(
        f"[bold]Target Brand:[/bold] {TARGET_BRAND}\n"
        f"[bold]Min Intent Confidence:[/bold] {MIN_INTENT_CONFIDENCE}\n"
        f"[bold]Min Retrieval Similarity:[/bold] {MIN_RETRIEVAL_SIMILARITY}\n"
        f"[bold]Frustration Threshold:[/bold] {FRUSTRATION_THRESHOLD}\n",
        title="[bold cyan]System Information[/bold cyan]",
        expand=False,
    )
    console.print(panel)


if __name__ == "__main__":
    app()
