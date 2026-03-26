"""Core data types for EmmentalEmbed."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class EmbeddingSet:
    """Container for extracted PLM embeddings.

    Attributes:
        embeddings: DataFrame with variant_id index and embedding dimensions as columns.
        model_name: HuggingFace model identifier used for extraction.
        layer: Hidden-state layer index that was extracted.
        representation: Pooling strategy used (mean, bos, per_token).
    """

    embeddings: pd.DataFrame
    model_name: str
    layer: int = -1
    representation: str = "mean"

    @property
    def variant_ids(self) -> list[str]:
        """Return list of variant IDs."""
        return list(self.embeddings.index)

    @property
    def dim(self) -> int:
        """Return embedding dimensionality."""
        return self.embeddings.shape[1]

    def __len__(self) -> int:
        """Return number of sequences."""
        return len(self.embeddings)

    def save(self, path: str | Path) -> None:
        """Save embeddings to parquet."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.embeddings.to_parquet(path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        model_name: str = "unknown",
        layer: int = -1,
        representation: str = "mean",
    ) -> EmbeddingSet:
        """Load embeddings from parquet."""
        df = pd.read_parquet(path)
        df.index.name = "variant_id"
        return cls(
            embeddings=df,
            model_name=model_name,
            layer=layer,
            representation=representation,
        )


@dataclass
class FoldResult:
    """Container for structure prediction results.

    Attributes:
        pdb_dir: Directory containing output PDB files.
        summary: DataFrame with per-sequence metrics (pLDDT, etc.).
        method: Folding method used (chai, af3).
        n_sequences: Number of sequences processed.
        n_failed: Number of sequences that failed.
    """

    pdb_dir: Path
    summary: pd.DataFrame
    method: str
    n_sequences: int = 0
    n_failed: int = 0

    def save_summary(self, path: str | Path) -> None:
        """Save summary metrics to CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.summary.to_csv(path, index=False)
