"""Structure prediction subpackage."""

from __future__ import annotations

from emmentalembed.config import FoldConfig
from emmentalembed.types import FoldResult


def fold_structures(config: FoldConfig) -> FoldResult:
    """Dispatch structure prediction to the appropriate method.

    Args:
        config: Fold configuration specifying method, paths, and parameters.

    Returns:
        FoldResult with output PDB directory and summary metrics.

    Raises:
        ValueError: If the method is not supported.
        ImportError: If the required package for the method is not installed.
    """
    if config.method == "chai":
        from emmentalembed.fold.chai import fold_chai

        return fold_chai(config)
    elif config.method == "af3":
        from emmentalembed.fold.af3 import fold_af3

        return fold_af3(config)
    else:
        raise ValueError(
            f"Unknown folding method: {config.method!r}. Supported: 'chai', 'af3'"
        )
