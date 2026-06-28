"""``llmstudio`` command-line interface.

Commands:
    llmstudio setup     # ensure workspace + download the assistant model
    llmstudio ui        # launch the web studio
    llmstudio doctor    # diagnose the environment
    llmstudio models    # list the base-model catalog
    llmstudio jobs      # list training jobs
    llmstudio datasets  # list prepared datasets
    llmstudio version
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from llmstudio.core.utils.logging import setup_logging
from llmstudio.version import __version__

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="LLM Studio — no-code fine-tuning for open-source LLMs.",
)
console = Console()


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging.")) -> None:
    setup_logging("DEBUG" if verbose else "INFO")


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"[bold]LLM Studio[/bold] v{__version__}")


@app.command()
def setup(
    assistant: bool = typer.Option(True, "--assistant/--no-assistant", help="Download the assistant model."),
) -> None:
    """Prepare the workspace and (optionally) download the assistant model."""
    from llmstudio.services import get_studio

    studio = get_studio()
    console.rule("[bold]LLM Studio setup")
    result = studio.system.setup(download_assistant=assistant, progress=lambda m: console.print(f"  • {m}"))
    console.rule("[green]Done")
    for key, value in result.items():
        console.print(f"[bold]{key}[/bold]: {value}")


@app.command()
def doctor() -> None:
    """Diagnose the environment (GPU, deps, paths, HF auth)."""
    from llmstudio.services import get_studio

    checks = get_studio().system.doctor()
    table = Table(title="Environment doctor")
    table.add_column("Check", style="bold")
    table.add_column("Status")
    table.add_column("Detail")
    for c in checks:
        table.add_row(c.name, "[green]OK[/green]" if c.ok else "[red]FAIL[/red]", c.detail)
    console.print(table)


@app.command()
def ui(
    host: Optional[str] = typer.Option(None, help="Bind address (default from config)."),
    port: Optional[int] = typer.Option(None, help="Port (default from config)."),
    share: bool = typer.Option(False, "--share", help="Create a public Gradio share link."),
) -> None:
    """Launch the web studio."""
    from llmstudio.ui.app import launch

    console.print(f"[bold]Launching LLM Studio[/bold] v{__version__} …")
    launch(host=host, port=port, share=share or None)


@app.command()
def models() -> None:
    """List the base-model catalog."""
    from llmstudio.core.models.catalog import ModelCatalog

    catalog = ModelCatalog.load()
    table = Table(title=f"Base models ({len(catalog.all())})")
    for col in ("Key", "Name", "Params", "Family", "Ctx", "Gated", "Tags"):
        table.add_column(col)
    for e in sorted(catalog.all(), key=lambda x: (not x.is_recommended, x.params_b)):
        table.add_row(
            e.key, e.name, f"{e.params_b:g}B", e.family, str(e.context_length),
            "yes" if e.gated else "no", ", ".join(e.tags),
        )
    console.print(table)


@app.command()
def jobs() -> None:
    """List training jobs and their state."""
    from llmstudio.services import get_studio

    js = get_studio().training.list_jobs()
    if not js:
        console.print("[dim]No jobs yet.[/dim]")
        return
    table = Table(title=f"Jobs ({len(js)})")
    for col in ("ID", "Name", "Status", "Model", "Step", "Progress"):
        table.add_column(col)
    for j in js:
        table.add_row(j.id, j.name or "-", j.status.value, j.base_model_key,
                      f"{j.current_step}/{j.total_steps or '?'}", f"{int(j.progress * 100)}%")
    console.print(table)


@app.command()
def datasets() -> None:
    """List prepared datasets."""
    from llmstudio.services import get_studio

    ds = get_studio().data.list_datasets()
    if not ds:
        console.print("[dim]No prepared datasets yet.[/dim]")
        return
    table = Table(title=f"Datasets ({len(ds)})")
    for col in ("ID", "Name", "Format", "Train", "Eval"):
        table.add_column(col)
    for d in ds:
        table.add_row(d.dataset_id, d.name, d.task_format.value, str(d.n_train), str(d.n_eval))
    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
