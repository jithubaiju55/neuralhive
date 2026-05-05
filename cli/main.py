"""
NeuralHive CLI — Main Entry Point
===================================
Run big AI models on weak hardware. Free. Forever. Offline.

Usage:
  neuralhive setup              — first time setup
  neuralhive "build me an app"  — build something
  neuralhive chat               — interactive chat mode
  neuralhive models             — list available models
  neuralhive status             — show system info
"""

import os
import sys
import json
import time
import click
from pathlib import Path
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.syntax import Syntax
    from rich.markdown import Markdown
    from rich.live import Live
    from rich.text import Text
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None

CONFIG_DIR = Path.home() / ".neuralhive"
CONFIG_FILE = CONFIG_DIR / "config.json"


def load_config() -> dict:
    try:
        if CONFIG_FILE.exists():
            return json.loads(CONFIG_FILE.read_text())
    except Exception:
        pass
    return {}


def save_config(config: dict):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2))


def print_banner():
    if RICH_AVAILABLE:
        banner = """[bold cyan]
 ███╗   ██╗███████╗██╗   ██╗██████╗  █████╗ ██╗      ██╗  ██╗██╗██╗   ██╗███████╗
 ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔══██╗██║      ██║  ██║██║██║   ██║██╔════╝
 ██╔██╗ ██║█████╗  ██║   ██║██████╔╝███████║██║      ███████║██║██║   ██║█████╗  
 ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██╔══██║██║      ██╔══██║██║╚██╗ ██╔╝██╔══╝  
 ██║ ╚████║███████╗╚██████╔╝██║  ██║██║  ██║███████╗ ██║  ██║██║ ╚████╔╝ ███████╗
 ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝  ╚══════╝
[/bold cyan]
[dim]Run massive AI models on weak hardware. Free. Forever. Offline.[/dim]"""
        console.print(banner)
    else:
        print("=" * 60)
        print("  NEURALHIVE — Free Local AI Coding Agent")
        print("  Run big models on weak hardware. Free. Forever.")
        print("=" * 60)


def get_engine(config: dict):
    """Load the inference engine with saved config."""
    from core.engine import InferenceEngine, InferenceConfig

    model_path = config.get("model_path")
    storage_path = config.get("storage_path", str(Path.home()))

    if not model_path or not Path(model_path).exists():
        if RICH_AVAILABLE:
            console.print("[red]❌ No model loaded. Run [bold]neuralhive setup[/bold] first.[/red]")
        else:
            print("❌ No model loaded. Run 'neuralhive setup' first.")
        sys.exit(1)

    engine_config = InferenceConfig(model_path=model_path)
    engine = InferenceEngine(engine_config, storage_path)

    if RICH_AVAILABLE:
        with console.status("[cyan]Loading model...[/cyan]"):
            success = engine.load()
    else:
        print("Loading model...")
        success = engine.load()

    if not success:
        sys.exit(1)

    return engine


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    """NeuralHive — Free local AI coding agent."""
    if ctx.invoked_subcommand is None:
        print_banner()
        click.echo(ctx.get_help())


@main.command()
def setup():
    """First-time setup: choose storage, download model."""
    print_banner()

    if RICH_AVAILABLE:
        console.print("\n[bold yellow]🔧 NeuralHive Setup[/bold yellow]\n")
    else:
        print("\n=== NeuralHive Setup ===\n")

    config = load_config()

    # Step 1: Detect storage
    from core.storage.detector import StorageDetector

    if RICH_AVAILABLE:
        console.print("[cyan]Step 1: Detecting storage devices...[/cyan]")
    else:
        print("Step 1: Detecting storage devices...")

    detector = StorageDetector()

    # Check for saved storage
    saved_path = detector.get_saved_path()
    if saved_path:
        if RICH_AVAILABLE:
            console.print(f"[green]✅ Previously configured storage: {saved_path}[/green]")
        else:
            print(f"✅ Using saved storage: {saved_path}")
        storage_path = saved_path
    else:
        devices = detector.detect_all()

        if RICH_AVAILABLE:
            table = Table(title="Available Storage", show_header=True)
            table.add_column("#", style="dim", width=3)
            table.add_column("Path", style="cyan")
            table.add_column("Type", style="green")
            table.add_column("Free", style="yellow")
            table.add_column("Speed", style="blue")
            table.add_column("Status")

            for i, d in enumerate(devices):
                rec = " ⭐ Recommended" if d.is_recommended else ""
                warn = f" ⚠️  {d.warning}" if d.warning else ""
                table.add_row(
                    str(i + 1),
                    d.label,
                    d.storage_type.value,
                    f"{d.free_gb:.0f} GB",
                    d.speed_rating,
                    f"{rec}{warn}" or "OK"
                )
            console.print(table)
        else:
            for i, d in enumerate(devices):
                rec = " [RECOMMENDED]" if d.is_recommended else ""
                print(f"  [{i+1}] {d.label} | {d.free_gb:.0f}GB free | {d.speed_rating}{rec}")

        default = next((i+1 for i, d in enumerate(devices) if d.is_recommended), 1)
        choice = click.prompt(
            f"\nChoose storage (1-{len(devices)})",
            default=str(default)
        )

        try:
            chosen = devices[int(choice) - 1]
            storage_path = chosen.path
            detector.save_choice(storage_path)
            if RICH_AVAILABLE:
                console.print(f"[green]✅ Storage set to: {storage_path}[/green]")
        except (IndexError, ValueError):
            storage_path = str(Path.home())

    # Step 2: Choose model
    from core.models.selector import ModelSelector, ModelDownloader, GGUFScanner

    # Define models_dir early — needed for local model scanning
    models_dir = Path(storage_path) / "neuralhive_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if RICH_AVAILABLE:
        console.print("\n[cyan]Step 2: Selecting model for your hardware...[/cyan]")
    else:
        print("\nStep 2: Selecting model for your hardware...")

    selector = ModelSelector()
    sys_info = selector.get_system_info()
    compatible = selector.get_compatible_models()
    recommended = selector.get_recommended()

    # Scan for manually placed .gguf files in models dir
    scanner = GGUFScanner(models_dir)
    local_models = scanner.scan()
    # Remove already-known models from local scan (avoid duplicates)
    known_filenames = {m.filename for m in compatible}
    extra_local = [m for m in local_models if m["filename"] not in known_filenames]

    if RICH_AVAILABLE:
        console.print(f"[dim]RAM: {sys_info['ram_gb']}GB | "
                     f"Effective with NeuralHive runtime: {sys_info['effective_ram_gb_with_runtime']}GB[/dim]")

        table = Table(title="Compatible Models", show_header=True)
        table.add_column("#", width=3)
        table.add_column("Model", style="cyan")
        table.add_column("Size", style="yellow")
        table.add_column("RAM Needed", style="green")
        table.add_column("Coding Score", style="blue")
        table.add_column("Speed")

        for i, m in enumerate(compatible):
            rec = " ⭐" if m == recommended else ""
            table.add_row(
                str(i + 1),
                f"{m.display_name}{rec}",
                f"{m.size_gb:.0f} GB",
                f"{m.ram_required_gb:.0f} GB",
                f"{m.coding_score}/100",
                f"~{m.speed_tps} tok/s"
            )
        # Show extra local models (manually downloaded)
        if extra_local:
            console.print("\n[yellow]📁 Also found in your models folder (manually downloaded):[/yellow]")
            for i, m in enumerate(extra_local):
                known_tag = "" if m["is_known"] else " [dim](unrecognised — will still work)[/dim]"
                console.print(f"  [{len(compatible)+i+1}] [cyan]{m['display_name']}[/cyan]  {m['size_gb']:.0f}GB  score:{m['coding_score']}/100{known_tag}")

        console.print(table)
    else:
        print(f"Your RAM: {sys_info['ram_gb']}GB")
        for i, m in enumerate(compatible):
            rec = " [RECOMMENDED]" if m == recommended else ""
            print(f"  [{i+1}] {m.display_name} | {m.size_gb:.0f}GB | Score: {m.coding_score}/100{rec}")
        for i, m in enumerate(extra_local):
            print(f"  [{len(compatible)+i+1}] {m['display_name']} (local) | {m['size_gb']:.0f}GB")

    total_choices = len(compatible) + len(extra_local)
    default_idx = compatible.index(recommended) + 1 if recommended else 1
    choice = click.prompt(
        f"\nChoose model (1-{total_choices})",
        default=str(default_idx)
    )

    try:
        idx = int(choice) - 1
        if idx < len(compatible):
            chosen_model = compatible[idx]
            chosen_local_path = None
        else:
            # User picked a manually downloaded local model
            local_pick = extra_local[idx - len(compatible)]
            model_path = local_pick["path"]
            config.update({
                "storage_path": storage_path,
                "model_path": model_path,
                "model_name": local_pick["filename"],
                "model_display_name": local_pick["display_name"],
                "setup_complete": True,
            })
            save_config(config)
            if RICH_AVAILABLE:
                console.print(Panel(
                    f"[bold green]✅ Setup Complete![/bold green]\n\n"
                    f"Model: [cyan]{local_pick['display_name']}[/cyan]\n"
                    f"Path:  [dim]{model_path}[/dim]\n\n"
                    f"[bold]Start coding:[/bold]\n"
                    f"  [yellow]nh \"build me a todo app\"[/yellow]\n"
                    f"  [yellow]neuralhive chat[/yellow]",
                    title="NeuralHive Ready", border_style="green"
                ))
            return
        chosen_model = compatible[int(choice) - 1]
    except (IndexError, ValueError):
        chosen_model = recommended or compatible[0]

    # Step 3: Download model
    models_dir = Path(storage_path) / "neuralhive_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    downloader = ModelDownloader(models_dir)

    if downloader.is_downloaded(chosen_model):
        if RICH_AVAILABLE:
            console.print(f"[green]✅ {chosen_model.display_name} already downloaded![/green]")
        else:
            print(f"✅ {chosen_model.display_name} already downloaded!")
        model_path = str(downloader.get_model_path(chosen_model))
    else:
        if RICH_AVAILABLE:
            console.print(f"\n[cyan]Downloading {chosen_model.display_name} ({chosen_model.size_gb:.0f}GB)...[/cyan]")
            console.print(f"[dim]URL: {downloader.get_download_url(chosen_model)}[/dim]")
            console.print("\n[yellow]Tip: You can also download manually from HuggingFace and place in:[/yellow]")
            console.print(f"[dim]{models_dir}[/dim]\n")
        else:
            print(f"\nDownloading {chosen_model.display_name}...")

        confirmed = click.confirm(
            f"Download {chosen_model.size_gb:.0f}GB model? (or download manually)",
            default=True
        )

        if confirmed:
            if RICH_AVAILABLE:
                from rich.progress import (
                    Progress, DownloadColumn, TransferSpeedColumn,
                    TimeRemainingColumn, BarColumn, TextColumn
                )
                with Progress(
                    TextColumn("[bold cyan]{task.description}"),
                    BarColumn(bar_width=40),
                    "[progress.percentage]{task.percentage:>3.0f}%",
                    "•",
                    DownloadColumn(),
                    "•",
                    TransferSpeedColumn(),
                    "•",
                    TimeRemainingColumn(),
                    console=console,
                    transient=False,
                ) as progress:
                    total_bytes = int(chosen_model.size_gb * 1024 ** 3)
                    task = progress.add_task(
                        f"Downloading {chosen_model.display_name}",
                        total=total_bytes
                    )
                    last_downloaded = [0]

                    def on_progress(downloaded, total):
                        delta = downloaded - last_downloaded[0]
                        if delta > 0:
                            progress.update(task, advance=delta, total=total)
                            last_downloaded[0] = downloaded

                    try:
                        model_path = str(downloader.download(chosen_model, on_progress))
                        progress.update(task, completed=total_bytes)
                    except Exception as e:
                        console.print(f"[red]❌ Download failed: {e}[/red]")
                        console.print(f"[yellow]Download manually from:[/yellow]")
                        console.print(f"[dim]{downloader.get_download_url(chosen_model)}[/dim]")
                        console.print(f"[yellow]Then place the file in: {models_dir}[/yellow]")
                        manual_path = click.prompt("\nEnter path to downloaded file (or Ctrl+C to exit)")
                        model_path = manual_path

                console.print("[bold green]✅ Download complete![/bold green]")
            else:
                # Fallback plain progress for non-rich environments
                last_pct = [0]

                def on_progress(downloaded, total):
                    pct = int(downloaded / total * 100)
                    gb_done = downloaded / (1024**3)
                    gb_total = total / (1024**3)
                    bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
                    print(f"\r  [{bar}] {pct:3d}%  {gb_done:.1f}/{gb_total:.1f} GB", end="", flush=True)
                    last_pct[0] = pct

                try:
                    model_path = str(downloader.download(chosen_model, on_progress))
                    print("\n✅ Download complete!")
                except Exception as e:
                    print(f"\n❌ Download failed: {e}")
                    sys.exit(1)
        else:
            manual_path = click.prompt("Enter path to manually downloaded model file")
            model_path = manual_path

    # Save config
    config.update({
        "storage_path": storage_path,
        "model_path": model_path,
        "model_name": chosen_model.name,
        "model_display_name": chosen_model.display_name,
        "setup_complete": True,
    })
    save_config(config)

    if RICH_AVAILABLE:
        console.print(Panel(
            f"[bold green]✅ Setup Complete![/bold green]\n\n"
            f"Model: [cyan]{chosen_model.display_name}[/cyan]\n"
            f"Storage: [cyan]{storage_path}[/cyan]\n\n"
            f"[bold]Start coding:[/bold]\n"
            f"  [yellow]neuralhive \"build me a todo app\"[/yellow]\n"
            f"  [yellow]neuralhive chat[/yellow]",
            title="NeuralHive Ready",
            border_style="green"
        ))
    else:
        print("\n✅ Setup complete!")
        print(f"Model: {chosen_model.display_name}")
        print("\nStart with: neuralhive \"build me an app\"")


@main.command()
@click.argument('prompt')
@click.option('--dir', '-d', default=None, help='Output directory for generated files')
@click.option('--model', '-m', default=None, help='Model to use (overrides config)')
def build(prompt, dir, model):
    """Build a complete app from a prompt."""
    config = load_config()

    if not config.get("setup_complete"):
        if RICH_AVAILABLE:
            console.print("[red]Run [bold]neuralhive setup[/bold] first![/red]")
        else:
            print("Run 'neuralhive setup' first!")
        sys.exit(1)

    # Generate clean project folder name from prompt
    project_name = prompt[:40].lower()
    project_name = "".join(c if c.isalnum() else "_" for c in project_name).strip("_")
    project_name = "_".join(filter(None, project_name.split("_")))  # remove double underscores

    # Set working directory
    if dir:
        working_dir = Path(dir)
    else:
        # Default output location: Desktop/NeuralHive_Projects/
        desktop = Path.home() / "Desktop"
        default_base = desktop / "NeuralHive_Projects" if desktop.exists() else Path.home() / "NeuralHive_Projects"
        default_path = default_base / project_name

        if RICH_AVAILABLE:
            console.print(f"\n[dim]Default output: [cyan]{default_path}[/cyan][/dim]")
            use_default = click.confirm("Build project here?", default=True)
            if use_default:
                working_dir = default_path
            else:
                custom = click.prompt("Enter folder path", default=str(Path.cwd() / project_name))
                working_dir = Path(custom)
        else:
            print(f"\nDefault output: {default_path}")
            use_default = click.confirm("Build project here?", default=True)
            working_dir = default_path if use_default else Path(
                click.prompt("Enter folder path", default=str(Path.cwd() / project_name))
            )

    working_dir.mkdir(parents=True, exist_ok=True)

    if RICH_AVAILABLE:
        console.print(Panel(
            f"[bold cyan]🚀 Building:[/bold cyan] {prompt}\n"
            f"[dim]Output: {working_dir}[/dim]",
            border_style="cyan"
        ))
    else:
        print(f"\n🚀 Building: {prompt}")
        print(f"Output: {working_dir}")

    engine = get_engine(config)

    from agent.loop import CodingAgent

    agent = CodingAgent(engine, str(working_dir))

    output_buffer = []

    def on_token(token, token_type):
        if token_type == "token":
            if RICH_AVAILABLE:
                console.print(token, end="", markup=False)
            else:
                print(token, end="", flush=True)
            output_buffer.append(token)
        elif token_type == "status":
            if RICH_AVAILABLE:
                console.print(token, style="bold yellow")
            else:
                print(token)

    result = agent.run(prompt, stream_callback=on_token)

    print()

    if RICH_AVAILABLE:
        status_color = "green" if result.success else "yellow"
        status_icon = "✅" if result.success else "⚠️"

        console.print(Panel(
            f"[bold {status_color}]{status_icon} {result.final_message}[/bold {status_color}]\n\n"
            f"[dim]Iterations: {result.iterations} | "
            f"Files: {len(result.files_created)}[/dim]\n\n"
            f"[bold]Open your project:[/bold]\n"
            f"  [yellow]cd {working_dir}[/yellow]",
            title="Build Complete",
            border_style=status_color
        ))
    else:
        print(f"\n{'✅' if result.success else '⚠️'} {result.final_message}")
        print(f"\nProject at: {working_dir}")


@main.command()
def chat():
    """Interactive chat mode — ask anything."""
    config = load_config()

    if not config.get("setup_complete"):
        print("Run 'neuralhive setup' first!")
        sys.exit(1)

    engine = get_engine(config)

    if RICH_AVAILABLE:
        console.print(Panel(
            "[bold cyan]NeuralHive Chat[/bold cyan]\n"
            "[dim]Type your message. 'quit' to exit. 'clear' to reset.[/dim]",
            border_style="cyan"
        ))
    else:
        print("\n=== NeuralHive Chat ===")
        print("Type 'quit' to exit\n")

    while True:
        try:
            if RICH_AVAILABLE:
                console.print("\n[bold green]You:[/bold green] ", end="")
            else:
                print("\nYou: ", end="")

            user_input = input().strip()

            if not user_input:
                continue
            if user_input.lower() in ('quit', 'exit', 'q'):
                break
            if user_input.lower() == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                continue

            if RICH_AVAILABLE:
                console.print("\n[bold cyan]NeuralHive:[/bold cyan]")
            else:
                print("\nNeuralHive:")

            for token in engine.generate_stream(user_input):
                if RICH_AVAILABLE:
                    console.print(token, end="", markup=False)
                else:
                    print(token, end="", flush=True)

            print()

        except KeyboardInterrupt:
            break
        except EOFError:
            break

    if RICH_AVAILABLE:
        console.print("\n[dim]Goodbye![/dim]")
    else:
        print("\nGoodbye!")


@main.command()
def models():
    """List all available models."""
    from core.models.selector import ModelSelector, AVAILABLE_MODELS, GGUFScanner

    config = load_config()
    selector = ModelSelector()
    sys_info = selector.get_system_info()
    compatible = selector.get_compatible_models()
    recommended = selector.get_recommended()

    # Scan for manually placed .gguf files in models dir
    storage_path = config.get("storage_path", str(Path.home()))
    models_dir = Path(storage_path) / "neuralhive_models"
    scanner = GGUFScanner(models_dir)
    local_models = scanner.scan()
    known_filenames = {m.filename for m in compatible}
    extra_local = [m for m in local_models if m["filename"] not in known_filenames]

    if RICH_AVAILABLE:
        console.print(Panel(
            f"[bold]System:[/bold] {sys_info['ram_gb']}GB RAM | "
            f"{sys_info['cpu_cores']} CPU cores\n"
            f"[dim]With NeuralHive runtime boost: {sys_info['effective_ram_gb_with_runtime']}GB effective[/dim]",
            title="Your System",
            border_style="cyan"
        ))

        table = Table(title="All Available Models", show_header=True, show_lines=True)
        table.add_column("Model", style="cyan", min_width=20)
        table.add_column("Size", style="yellow")
        table.add_column("RAM", style="green")
        table.add_column("Code", style="blue")
        table.add_column("Speed")
        table.add_column("Status")

        for m in AVAILABLE_MODELS:
            is_compat = m in compatible
            is_rec = m == recommended
            status = "⭐ Recommended" if is_rec else ("✅ Compatible" if is_compat else "❌ Needs more RAM")
            style = "green" if is_compat else "red"

            table.add_row(
                m.display_name,
                f"{m.size_gb:.0f}GB",
                f"{m.ram_required_gb:.0f}GB",
                f"{m.coding_score}/100",
                f"~{m.speed_tps} t/s",
                f"[{style}]{status}[/{style}]"
            )

        console.print(table)
    else:
        print(f"\nYour RAM: {sys_info['ram_gb']}GB")
        print("\nAll Models:")
        from core.models.selector import AVAILABLE_MODELS
        for m in AVAILABLE_MODELS:
            compat = "✅" if m in compatible else "❌"
            print(f"  {compat} {m.display_name} | {m.size_gb:.0f}GB | Score: {m.coding_score}/100")


@main.command()
def status():
    """Show system information and NeuralHive status."""
    import psutil
    config = load_config()

    mem = psutil.virtual_memory()
    cpu_count = psutil.cpu_count(logical=False)
    disk_partitions = psutil.disk_partitions()

    if RICH_AVAILABLE:
        console.print(Panel(
            f"[bold cyan]System Information[/bold cyan]\n\n"
            f"RAM Total:     [green]{mem.total / (1024**3):.1f}GB[/green]\n"
            f"RAM Available: [green]{mem.available / (1024**3):.1f}GB[/green]\n"
            f"RAM Used:      [yellow]{mem.percent}%[/yellow]\n"
            f"CPU Cores:     [green]{cpu_count}[/green]\n\n"
            f"[bold]NeuralHive Config:[/bold]\n"
            f"Setup:   [{'green]✅ Complete' if config.get('setup_complete') else 'red]❌ Not configured'}[/{'green' if config.get('setup_complete') else 'red'}]\n"
            f"Model:   [cyan]{config.get('model_display_name', 'None')}[/cyan]\n"
            f"Storage: [cyan]{config.get('storage_path', 'None')}[/cyan]",
            title="NeuralHive Status",
            border_style="cyan"
        ))
    else:
        print(f"\nRAM: {mem.total/(1024**3):.1f}GB total, {mem.available/(1024**3):.1f}GB available")
        print(f"CPU: {cpu_count} cores")
        print(f"Setup: {'✅ Complete' if config.get('setup_complete') else '❌ Not configured'}")
        print(f"Model: {config.get('model_display_name', 'None')}")



@click.command()
@click.argument('prompt')
@click.option('--dir', '-d', default=None, help='Output directory for the project')
def nh_command(prompt, dir):
    """Short alias — nh \"build me a todo app\" """
    from click import Context
    ctx = click.Context(build)
    with ctx:
        ctx.invoke(build, prompt=prompt, dir=dir, model=None)


if __name__ == "__main__":
    main()