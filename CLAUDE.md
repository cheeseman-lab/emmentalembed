# CLAUDE.md

Instructions for Claude Code when working in this repository.

## Project

**EmmentalEmbed** v0.1.0 — Protein embedding extraction and structure prediction at scale. Extracts embeddings from any HuggingFace-compatible protein language model, and predicts structures using Chai-1 or AlphaFold3.

## Development

```bash
eval "$(conda shell.bash hook)" && conda activate emmentalembed
uv pip install -e ".[dev]"        # base + tests
uv pip install -e ".[plm]"        # + torch/transformers for embedding
```

**Folding environments** are separate (dependency conflicts):
```bash
bash scripts/setup_fold_env.sh chai   # Chai-1
bash scripts/setup_fold_env.sh af3    # AlphaFold3 (weights required)
```

## Code Structure

```
emmentalembed/
├── src/emmentalembed/
│   ├── __init__.py         # Version, public API
│   ├── cli.py              # Typer CLI (embed, fold, version)
│   ├── config.py           # YAML-backed dataclass configs
│   ├── extract.py          # Unified PLM embedding extraction
│   ├── types.py            # EmbeddingSet, FoldResult
│   ├── io.py               # FASTA reading, output helpers
│   └── fold/
│       ├── __init__.py     # fold() dispatcher
│       ├── chai.py         # Chai-1 wrapper (ESM mode)
│       └── af3.py          # AlphaFold3 wrapper
├── scripts/
│   ├── embed.sh            # SLURM: embedding extraction
│   ├── fold_chai.sh        # SLURM: Chai-1 on A100
│   ├── fold_af3.sh         # SLURM: AF3 on A100
│   └── setup_fold_env.sh   # Creates fold conda envs
└── tests/
    ├── test_install.py     # Smoke: import, version, CLI, deps
    └── test_config.py      # Config roundtrip, defaults
```

## CLI Commands

```bash
emmentalembed embed --fasta input.fasta --model facebook/esm2_t33_650M_UR50D --output emb.csv
emmentalembed fold --fasta input.fasta --method chai --output-dir pdb/
emmentalembed version
```

## Supported PLMs

Any HuggingFace-compatible protein model: ESM-2 (650M/3B/15B), ESM-C, AMPLIFY, ProtT5, ANKH, ProtBERT, and more.

## Key Design Decisions

1. **Unified extraction**: Single `extract_embeddings()` handles all HF models with auto-detection of quirks
2. **Separate fold envs**: Chai-1 and AF3 have conflicting deps, each gets its own conda env
3. **AF3 weights not bundled**: Users must download their own (not redistributable)
4. **Chai-1 ESM mode**: Default — avoids MSA generation, nearly AF3 quality
5. **No ESMFold**: Found to be buggy, removed in favor of Chai-1/AF3

## Running Tests

```bash
python -m pytest tests/ -v
```

## Code Style

- Build: hatchling
- Linter: ruff (line-length 100)
- Docstrings: Google style
- Python: >=3.10
