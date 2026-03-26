"""Chai-1 structure prediction wrapper.

Chai-1 is recommended when you need evolutionary information but don't want to
generate MSAs. Its ESM mode uses ESM embeddings instead, giving AF3-like quality
with much simpler setup.

Requires: pip install chai_lab
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from emmentalembed.config import FoldConfig
from emmentalembed.io import read_fasta, write_pdb
from emmentalembed.types import FoldResult

logger = logging.getLogger(__name__)


def fold_chai(config: FoldConfig) -> FoldResult:
    """Predict structures using Chai-1.

    Args:
        config: FoldConfig with fasta_path, output_dir, and parameters.

    Returns:
        FoldResult with PDB directory and per-sequence metrics.

    Raises:
        ImportError: If chai_lab is not installed.
    """
    try:
        from chai_lab.chai1 import run_inference
    except ImportError as exc:
        raise ImportError(
            "Chai-1 requires the 'chai_lab' package.\n"
            "Install with: pip install chai_lab"
        ) from exc

    import torch

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
    checkpoint_csv = output_dir / "chai_checkpoint.csv"
    completed_ids: set[str] = set()
    if checkpoint_csv.exists():
        checkpoint_df = pd.read_csv(checkpoint_csv)
        completed_ids = set(checkpoint_df["sequence_id"].tolist())
        logger.info("Resuming from checkpoint: %d already completed", len(completed_ids))

    results: list[dict] = []
    if checkpoint_csv.exists():
        results = pd.read_csv(checkpoint_csv).to_dict("records")

    n_failed = 0

    for i, (seq_id, sequence) in enumerate(filtered.items()):
        if seq_id in completed_ids:
            continue

        try:
            # Chai-1 expects a FASTA-like input; write a temporary single-sequence file
            tmp_fasta = output_dir / f".tmp_{seq_id}.fasta"
            tmp_fasta.write_text(f">{seq_id}\n{sequence}\n")

            output = run_inference(
                fasta_file=tmp_fasta,
                output_dir=output_dir / seq_id,
                num_trunk_recycles=config.num_recycles,
                device=torch.device(config.device),
                use_esm_embeddings=config.use_esm_embeddings,
            )

            # Extract metrics from output
            plddt_mean = float("nan")
            if hasattr(output, "ptm"):
                plddt_mean = float(output.ptm)

            results.append({
                "sequence_id": seq_id,
                "sequence_length": len(sequence),
                "plddt_mean": plddt_mean,
                "status": "success",
            })

            # Clean up temp file
            tmp_fasta.unlink(missing_ok=True)

        except torch.cuda.OutOfMemoryError:
            logger.warning("OOM for %s (len=%d), skipping", seq_id, len(sequence))
            torch.cuda.empty_cache()
            results.append({
                "sequence_id": seq_id,
                "sequence_length": len(sequence),
                "plddt_mean": float("nan"),
                "status": "oom",
            })
            n_failed += 1

        except Exception as exc:
            logger.warning("Failed for %s: %s", seq_id, exc)
            results.append({
                "sequence_id": seq_id,
                "sequence_length": len(sequence),
                "plddt_mean": float("nan"),
                "status": f"error: {exc}",
            })
            n_failed += 1

        # Checkpoint
        if (i + 1) % config.checkpoint_interval == 0:
            pd.DataFrame(results).to_csv(checkpoint_csv, index=False)
            logger.info("Checkpoint saved: %d/%d complete", len(results), len(filtered))

    # Final save
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(output_dir / "fold_summary.csv", index=False)
    checkpoint_csv.unlink(missing_ok=True)

    return FoldResult(
        pdb_dir=output_dir,
        summary=summary_df,
        method="chai",
        n_sequences=len(filtered),
        n_failed=n_failed,
    )
