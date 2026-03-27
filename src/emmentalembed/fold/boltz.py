"""Boltz-2 structure prediction wrapper.

Boltz-2 is an open-source (MIT) structure prediction model from MIT/Recursion.
Supports proteins, DNA, RNA, ligands, and binding affinity prediction.

Requires: pip install boltz[cuda]
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from emmentalembed.config import FoldConfig
from emmentalembed.io import read_fasta
from emmentalembed.types import FoldResult

logger = logging.getLogger(__name__)


def fold_boltz(config: FoldConfig) -> FoldResult:
    """Predict structures using Boltz-2.

    Args:
        config: FoldConfig with fasta_path, output_dir, and parameters.

    Returns:
        FoldResult with output directory and per-sequence metrics.

    Raises:
        ImportError: If boltz is not installed.
        RuntimeError: If boltz predict fails.
    """
    if shutil.which("boltz") is None:
        raise ImportError(
            "Boltz-2 requires the 'boltz' package.\n"
            "Install with: uv pip install 'boltz[cuda]'"
        )

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
    checkpoint_csv = output_dir / "boltz_checkpoint.csv"
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
            # Write single-sequence YAML (preferred Boltz-2 input format)
            seq_dir = output_dir / seq_id
            seq_dir.mkdir(parents=True, exist_ok=True)
            tmp_yaml = seq_dir / f"{seq_id}.yaml"
            tmp_yaml.write_text(
                f"version: 1\nsequences:\n  - protein:\n"
                f"      id: A\n      sequence: {sequence}\n"
            )

            # Run boltz predict
            cmd = [
                "boltz", "predict",
                str(tmp_yaml),
                "--out_dir", str(seq_dir),
                "--recycling_steps", str(config.num_recycles),
                "--diffusion_samples", "1",
                "--output_format", "pdb",
            ]

            # Boltz-2 requires MSAs; use the MSA server by default
            cmd.append("--use_msa_server")
            # Disable custom CUDA kernels (cuequivariance) for compatibility
            cmd.append("--no_kernels")

            logger.info("Running Boltz-2 for %s (len=%d)", seq_id, len(sequence))
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,
            )

            if proc.returncode != 0:
                raise RuntimeError(
                    f"boltz predict failed (rc={proc.returncode}):\n"
                    f"stdout: {proc.stdout[-300:]}\nstderr: {proc.stderr[-300:]}"
                )

            if proc.stderr:
                logger.info("Boltz stderr for %s: %s", seq_id, proc.stderr[:300])

            # Parse confidence scores from output
            # Boltz outputs to boltz_results_<stem>/ inside out_dir
            plddt_mean = float("nan")
            ptm = float("nan")

            # Check for prediction files (CIF/PDB + confidence)
            pred_files = list(seq_dir.rglob("*.pdb")) + list(seq_dir.rglob("*.cif"))
            conf_files = list(seq_dir.rglob("confidence_*.json"))

            if not pred_files:
                logger.warning(
                    "No structure files produced for %s. Boltz stdout: %s",
                    seq_id, proc.stdout[:300],
                )

            if conf_files:
                with open(conf_files[0]) as f:
                    conf = json.load(f)
                plddt_mean = conf.get("complex_plddt", float("nan"))
                ptm = conf.get("ptm", float("nan"))
            else:
                # Try reading pLDDT from npz files
                plddt_files = list(seq_dir.rglob("plddt_*.npz"))
                if plddt_files:
                    import numpy as np
                    data = np.load(plddt_files[0])
                    arr = data[list(data.keys())[0]]
                    plddt_mean = float(arr.mean())

            results.append({
                "sequence_id": seq_id,
                "sequence_length": len(sequence),
                "plddt_mean": plddt_mean,
                "ptm": ptm,
                "status": "success",
            })

            # Clean up temp yaml
            tmp_yaml.unlink(missing_ok=True)

        except subprocess.TimeoutExpired:
            logger.warning("Timeout for %s (len=%d), skipping", seq_id, len(sequence))
            results.append({
                "sequence_id": seq_id,
                "sequence_length": len(sequence),
                "plddt_mean": float("nan"),
                "ptm": float("nan"),
                "status": "timeout",
            })
            n_failed += 1

        except Exception as exc:
            logger.warning("Failed for %s: %s", seq_id, exc)
            results.append({
                "sequence_id": seq_id,
                "sequence_length": len(sequence),
                "plddt_mean": float("nan"),
                "ptm": float("nan"),
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
        method="boltz",
        n_sequences=len(filtered),
        n_failed=n_failed,
    )
