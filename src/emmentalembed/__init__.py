"""EmmentalEmbed: Protein embedding extraction and structure prediction at scale."""

__version__ = "0.1.0"

from emmentalembed.config import EmbedConfig, EmmentalConfig, FoldConfig
from emmentalembed.io import csv_to_fasta, read_fasta
from emmentalembed.types import EmbeddingSet, FoldResult

__all__ = [
    "EmbedConfig",
    "EmbeddingSet",
    "EmmentalConfig",
    "FoldConfig",
    "FoldResult",
    "csv_to_fasta",
    "read_fasta",
]
