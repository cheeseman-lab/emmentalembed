"""I/O utilities for FASTA files and results."""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def _uppercase_preserve_tokens(seq: str) -> str:
    """Uppercase amino acid characters while preserving special tokens like <eos>."""
    parts = re.split(r"(<[^>]+>)", seq)
    return "".join(part if part.startswith("<") else part.upper() for part in parts)


def read_fasta(path: str | Path) -> dict[str, str]:
    """Read a FASTA file and return a mapping of variant_id -> sequence.

    Args:
        path: Path to the FASTA file.

    Returns:
        Ordered dictionary mapping sequence IDs to uppercase amino acid sequences,
        with special tokens (e.g. ``<eos>``) preserved in their original case.

    Raises:
        FileNotFoundError: If the FASTA file does not exist.
    """
    from Bio import SeqIO

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {path}")

    sequences: dict[str, str] = {}
    for record in SeqIO.parse(str(path), "fasta"):
        sequences[record.id] = _uppercase_preserve_tokens(str(record.seq))

    logger.info("Read %d sequences from %s", len(sequences), path)
    return sequences


def csv_to_fasta(
    csv_path: str | Path,
    fasta_path: str | Path,
    id_col: str = "id",
    seq_col: str = "sequence",
) -> int:
    """Convert a CSV with protein IDs and sequences to FASTA format.

    Args:
        csv_path: Path to input CSV file.
        fasta_path: Path for output FASTA file.
        id_col: Column name for sequence identifiers.
        seq_col: Column name for amino acid sequences.

    Returns:
        Number of sequences written.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        KeyError: If the required columns are not found.
    """
    import pandas as pd

    csv_path = Path(csv_path)
    fasta_path = Path(fasta_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    for col in (id_col, seq_col):
        if col not in df.columns:
            raise KeyError(f"Column {col!r} not found in CSV. Available: {list(df.columns)}")

    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fasta_path, "w") as f:
        for _, row in df.iterrows():
            seq_id = str(row[id_col]).strip()
            seq = str(row[seq_col]).strip()
            f.write(f">{seq_id}\n{seq}\n")

    logger.info("Wrote %d sequences to %s", len(df), fasta_path)
    return len(df)


def write_pdb(pdb_string: str, path: str | Path) -> None:
    """Write a PDB string to a file.

    Args:
        pdb_string: PDB-format string content.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pdb_string)
