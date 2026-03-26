"""Smoke tests: does the package install and import correctly?"""

import subprocess


def test_import():
    """Package imports without error."""
    import emmentalembed  # noqa: F401


def test_version():
    """Version string is set and follows semver."""
    from emmentalembed import __version__

    assert __version__
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_cli_entry_point():
    """CLI entry point is registered and callable."""
    result = subprocess.run(["emmentalembed", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "embed" in result.stdout.lower()
    assert "fold" in result.stdout.lower()
    assert "version" in result.stdout.lower()


def test_core_dependencies_importable():
    """All core dependencies are importable."""
    import yaml  # noqa: F401
    import typer  # noqa: F401
    import rich  # noqa: F401
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import tqdm  # noqa: F401
    import Bio  # noqa: F401


def test_public_api():
    """Public API types are accessible."""
    from emmentalembed import (
        EmbedConfig,
        EmbeddingSet,
        EmmentalConfig,
        FoldConfig,
        FoldResult,
        read_fasta,
    )

    assert callable(read_fasta)
    assert EmbedConfig is not None
    assert EmbeddingSet is not None
    assert EmmentalConfig is not None
    assert FoldConfig is not None
    assert FoldResult is not None
