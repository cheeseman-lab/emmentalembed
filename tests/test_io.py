"""Tests for FASTA I/O and CSV conversion."""

from pathlib import Path

import pytest

from emmentalembed.io import csv_to_fasta, read_fasta


@pytest.fixture
def example_csv():
    return Path(__file__).resolve().parent.parent / "data" / "example.csv"


@pytest.fixture
def example_fasta():
    return Path(__file__).resolve().parent.parent / "data" / "example.fasta"


def test_read_fasta(example_fasta):
    """Read example FASTA and verify contents."""
    seqs = read_fasta(example_fasta)
    assert len(seqs) == 5
    assert "GFP" in seqs
    assert "insulin_A" in seqs
    assert "ubiquitin" in seqs
    assert seqs["insulin_A"] == "GIVEQCCTSICSLYQLENYCN"
    assert len(seqs["GFP"]) == 238


def test_read_fasta_missing():
    """Reading a nonexistent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        read_fasta("/nonexistent/path.fasta")


def test_csv_to_fasta(example_csv, tmp_path):
    """Convert CSV to FASTA and read it back."""
    out = tmp_path / "converted.fasta"
    n = csv_to_fasta(example_csv, out)
    assert n == 5
    assert out.exists()

    # Read back and verify round-trip
    seqs = read_fasta(out)
    assert len(seqs) == 5
    assert "GFP" in seqs
    assert seqs["insulin_B"] == "FVNQHLCGSHLVEALYLVCGERGFFYTPKT"


def test_csv_to_fasta_custom_columns(tmp_path):
    """CSV conversion works with custom column names."""
    csv_path = tmp_path / "custom.csv"
    csv_path.write_text("protein_name,aa_seq\ntest_protein,ACDEFGHIKLMNPQRSTVWY\n")

    out = tmp_path / "custom.fasta"
    n = csv_to_fasta(csv_path, out, id_col="protein_name", seq_col="aa_seq")
    assert n == 1

    seqs = read_fasta(out)
    assert seqs["test_protein"] == "ACDEFGHIKLMNPQRSTVWY"


def test_csv_to_fasta_missing_column(tmp_path):
    """Missing column raises KeyError."""
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("name,seq\nfoo,ACDE\n")

    with pytest.raises(KeyError, match="id"):
        csv_to_fasta(csv_path, tmp_path / "out.fasta")


def test_csv_to_fasta_missing_file(tmp_path):
    """Missing CSV file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        csv_to_fasta(tmp_path / "nope.csv", tmp_path / "out.fasta")
