"""YAML-backed configuration dataclasses for embedding and folding jobs."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class EmbedConfig:
    """Configuration for PLM embedding extraction.

    Attributes:
        fasta_path: Path to input FASTA file.
        output_path: Path for output CSV.
        model: HuggingFace model identifier.
        batch_size: Sequences per forward pass.
        dtype: Model precision (float16, bfloat16, float32).
        layer: Hidden-state layer index (-1 = last).
        representation: Pooling strategy (mean, bos, per_token).
        device: PyTorch device string.
        device_map: HuggingFace device map for model sharding.
    """

    fasta_path: str = ""
    output_path: str = "embeddings.parquet"
    model: str = "facebook/esm2_t33_650M_UR50D"
    batch_size: int = 32
    dtype: str = "float16"
    layer: int = -1
    representation: str = "mean"
    device: str = "cuda"
    device_map: Optional[str] = None


@dataclass
class FoldConfig:
    """Configuration for structure prediction.

    Attributes:
        fasta_path: Path to input FASTA file.
        output_dir: Directory for output PDB files.
        method: Folding method (chai, boltz).
        max_seq_len: Maximum sequence length to process.
        num_recycles: Number of recycling iterations.
        checkpoint_interval: Save progress every N proteins.
        device: PyTorch device string.
        use_esm_embeddings: For Chai-1, use ESM mode instead of MSAs.
    """

    fasta_path: str = ""
    output_dir: str = "pdb_output"
    method: str = "chai"
    max_seq_len: int = 1024
    num_recycles: int = 3
    checkpoint_interval: int = 50
    device: str = "cuda"
    use_esm_embeddings: bool = True
    use_msa_server: bool = False


@dataclass
class EmmentalConfig:
    """Top-level configuration wrapping embed and fold settings.

    Attributes:
        embed: Embedding extraction configuration.
        fold: Structure prediction configuration.
    """

    embed: EmbedConfig = field(default_factory=EmbedConfig)
    fold: FoldConfig = field(default_factory=FoldConfig)

    def to_yaml(self, path: str | Path) -> None:
        """Serialize configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> EmmentalConfig:
        """Load configuration from a YAML file."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        embed_data = data.get("embed", {})
        fold_data = data.get("fold", {})

        return cls(
            embed=EmbedConfig(**{k: v for k, v in embed_data.items() if v is not None}),
            fold=FoldConfig(**{k: v for k, v in fold_data.items() if v is not None}),
        )
