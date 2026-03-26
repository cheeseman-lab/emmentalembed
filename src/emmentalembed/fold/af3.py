"""AlphaFold3 structure prediction wrapper.

AF3 is the gold standard for structure prediction. Use it when you have
templates or don't need MSAs.

Requires:
  - AF3 weights downloaded from DeepMind (not redistributable)
  - Request access: https://forms.gle/svvpY4u2jsHEwWYS6
  - alphafold3 package installed
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from emmentalembed.config import FoldConfig
from emmentalembed.io import read_fasta
from emmentalembed.types import FoldResult

logger = logging.getLogger(__name__)


def fold_af3(config: FoldConfig) -> FoldResult:
    """Predict structures using AlphaFold3.

    Args:
        config: FoldConfig with fasta_path, output_dir, weights_path, and parameters.

    Returns:
        FoldResult with PDB directory and per-sequence metrics.

    Raises:
        FileNotFoundError: If weights_path does not exist.
        ImportError: If alphafold3 dependencies are not installed.
    """
    weights_path = Path(config.weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"AF3 weights not found at: {weights_path}\n"
            "You must download your own copy of the AlphaFold3 weights.\n"
            "Request access: https://forms.gle/svvpY4u2jsHEwWYS6\n"
            "Then set weights_path in your config or pass --weights to the CLI."
        )

    try:
        import torch
    except ImportError as exc:
        raise ImportError("AF3 requires PyTorch. Install with: pip install torch") from exc

    sequences = read_fasta(config.fasta_path)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter by max sequence length
    filtered = {
        sid: seq
        for sid, seq in sequences.items()
        if len(seq) <= config.max_seq_len
    }
    skipped = len(sequences) - len(filtered)
    if skipped > 0:
        logger.warning(
            "Skipping %d sequences exceeding max_seq_len=%d",
            skipped,
            config.max_seq_len,
        )

    # Check for existing checkpoint
    checkpoint_csv = output_dir / "af3_checkpoint.csv"
    completed_ids: set[str] = set()
    if checkpoint_csv.exists():
        checkpoint_df = pd.read_csv(checkpoint_csv)
        completed_ids = set(checkpoint_df["sequence_id"].tolist())
        logger.info("Resuming from checkpoint: %d already completed", len(completed_ids))

    results: list[dict] = []
    if checkpoint_csv.exists():
        results = pd.read_csv(checkpoint_csv).to_dict("records")

    n_failed = 0

    # TODO: Implement AF3 inference loop once installation method is finalized.
    # The interface will follow the same pattern as chai.py:
    # - Iterate over sequences
    # - Call AF3 inference per sequence
    # - Extract pLDDT and save PDB
    # - Checkpoint every N proteins
    # - Handle OOM gracefully

    logger.warning(
        "AF3 inference is not yet fully implemented. "
        "Weights found at %s. Awaiting AF3 package installation instructions.",
        weights_path,
    )

    summary_df = pd.DataFrame(results)
    if not summary_df.empty:
        summary_df.to_csv(output_dir / "fold_summary.csv", index=False)

    return FoldResult(
        pdb_dir=output_dir,
        summary=summary_df,
        method="af3",
        n_sequences=len(filtered),
        n_failed=n_failed,
    )
