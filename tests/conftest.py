"""Shared fixtures for EmmentalEmbed tests."""

from pathlib import Path

import pytest


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def tmp_config(tmp_path):
    """Create a minimal config YAML for testing."""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(
        "embed:\n"
        "  fasta_path: input.fasta\n"
        "  output_path: embeddings.csv\n"
        "  model: facebook/esm2_t33_650M_UR50D\n"
        "fold:\n"
        "  fasta_path: input.fasta\n"
        "  output_dir: pdb_output\n"
        "  method: chai\n"
    )
    return config_path
