"""CLI entry point for EmmentalEmbed."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint

app = typer.Typer(
    name="emmentalembed",
    help="Protein embedding extraction and structure prediction at scale.",
    no_args_is_help=True,
)


@app.command()
def embed(
    fasta: Path = typer.Option(..., "--fasta", "-f", help="Path to input FASTA file"),
    model: str = typer.Option(
        "facebook/esm2_t33_650M_UR50D",
        "--model",
        "-m",
        help="HuggingFace model ID",
    ),
    output: Path = typer.Option(
        "embeddings.parquet",
        "--output",
        "-o",
        help="Output CSV path",
    ),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size"),
    dtype: str = typer.Option("float16", "--dtype", help="Model precision"),
    layer: int = typer.Option(-1, "--layer", "-l", help="Hidden layer index (-1 = last)"),
    representation: str = typer.Option(
        "mean",
        "--representation",
        "-r",
        help="Pooling strategy: mean, bos, per_token",
    ),
    device: str = typer.Option("cuda", "--device", help="PyTorch device"),
    device_map: Optional[str] = typer.Option(
        None, "--device-map", help="HuggingFace device map (e.g. auto)"
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="YAML config file (overrides other flags)"
    ),
) -> None:
    """Extract PLM embeddings from protein sequences."""
    try:
        from emmentalembed.extract import extract_embeddings
    except ImportError:
        rprint("[red]Error: PLM dependencies not installed.[/red]")
        rprint("Install with: [bold]pip install emmentalembed\\[plm][/bold]")
        raise typer.Exit(1)

    if config is not None:
        from emmentalembed.config import EmmentalConfig

        cfg = EmmentalConfig.from_yaml(config)
        fasta = Path(cfg.embed.fasta_path) if cfg.embed.fasta_path else fasta
        output = Path(cfg.embed.output_path) if cfg.embed.output_path else output
        model = cfg.embed.model
        batch_size = cfg.embed.batch_size
        dtype = cfg.embed.dtype
        layer = cfg.embed.layer
        representation = cfg.embed.representation
        device = cfg.embed.device
        device_map = cfg.embed.device_map

    if not fasta.exists():
        rprint(f"[red]Error: FASTA file not found: {fasta}[/red]")
        raise typer.Exit(1)

    rprint(f"[bold]Extracting embeddings[/bold]")
    rprint(f"  Model: {model}")
    rprint(f"  Input: {fasta}")
    rprint(f"  Output: {output}")

    result = extract_embeddings(
        fasta_path=fasta,
        model_id=model,
        output_path=output,
        layer=layer,
        representation=representation,
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        device_map=device_map,
    )
    rprint(f"[green]Done:[/green] {len(result)} sequences, dim={result.dim} -> {output}")


@app.command()
def fold(
    fasta: Path = typer.Option(..., "--fasta", "-f", help="Path to input FASTA file"),
    method: str = typer.Option("chai", "--method", "-m", help="Folding method: chai, af3"),
    output_dir: Path = typer.Option(
        "pdb_output", "--output-dir", "-o", help="Output directory for PDB files"
    ),
    weights_path: Optional[str] = typer.Option(
        None, "--weights", "-w", help="Path to model weights (required for AF3)"
    ),
    max_seq_len: int = typer.Option(1024, "--max-seq-len", help="Max sequence length"),
    num_recycles: int = typer.Option(3, "--num-recycles", help="Number of recycling iterations"),
    device: str = typer.Option("cuda", "--device", help="PyTorch device"),
    use_esm: bool = typer.Option(True, "--esm/--no-esm", help="Chai-1: use ESM mode (no MSAs)"),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="YAML config file (overrides other flags)"
    ),
) -> None:
    """Predict protein structures using Chai-1 or AlphaFold3."""
    from emmentalembed.fold import fold_structures

    if config is not None:
        from emmentalembed.config import EmmentalConfig

        cfg = EmmentalConfig.from_yaml(config)
        fasta = Path(cfg.fold.fasta_path) if cfg.fold.fasta_path else fasta
        output_dir = Path(cfg.fold.output_dir) if cfg.fold.output_dir else output_dir
        method = cfg.fold.method
        weights_path = cfg.fold.weights_path or weights_path
        max_seq_len = cfg.fold.max_seq_len
        num_recycles = cfg.fold.num_recycles
        device = cfg.fold.device
        use_esm = cfg.fold.use_esm_embeddings

    if not fasta.exists():
        rprint(f"[red]Error: FASTA file not found: {fasta}[/red]")
        raise typer.Exit(1)

    if method == "af3" and not weights_path:
        rprint("[red]Error: AF3 requires --weights path to model weights.[/red]")
        rprint("Request weights from DeepMind: https://forms.gle/svvpY4u2jsHEwWYS6")
        raise typer.Exit(1)

    rprint(f"[bold]Predicting structures[/bold]")
    rprint(f"  Method: {method}")
    rprint(f"  Input: {fasta}")
    rprint(f"  Output: {output_dir}")

    from emmentalembed.config import FoldConfig

    fold_config = FoldConfig(
        fasta_path=str(fasta),
        output_dir=str(output_dir),
        method=method,
        weights_path=weights_path or "",
        max_seq_len=max_seq_len,
        num_recycles=num_recycles,
        device=device,
        use_esm_embeddings=use_esm,
    )

    result = fold_structures(fold_config)
    rprint(
        f"[green]Done:[/green] {result.n_sequences} structures predicted "
        f"({result.n_failed} failed) -> {output_dir}"
    )


@app.command()
def version() -> None:
    """Print version information."""
    from emmentalembed import __version__

    rprint(f"emmentalembed v{__version__}")

    try:
        import torch

        rprint(f"PyTorch: {torch.__version__}")
        rprint(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            rprint(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        rprint("PyTorch: not installed")

    try:
        import transformers

        rprint(f"Transformers: {transformers.__version__}")
    except ImportError:
        rprint("Transformers: not installed")


if __name__ == "__main__":
    app()
