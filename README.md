# EmmentalEmbed

Protein embedding extraction and structure prediction at scale on the Whitehead HPC.

## What it does

1. **PLM Embeddings**: Extract embeddings from any HuggingFace protein language model (ESM-2, ProtT5, ANKH, AMPLIFY, ESM-C, etc.)
2. **Structure Prediction**: Predict 3D structures using Chai-1 or Boltz-2

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/cheeseman-lab/emmentalembed.git
cd emmentalembed

conda create -n emmentalembed -c conda-forge python=3.12 uv pip -y
conda activate emmentalembed
uv pip install -e ".[dev]"   # includes embeddings, folding, and test deps
```

### 2. Extract embeddings

```bash
# Interactive
emmentalembed embed --fasta proteins.fasta --model facebook/esm2_t33_650M_UR50D --output emb.parquet

# SLURM batch job
sbatch scripts/embed.sh
```

### 3. Predict structures (optional)

```bash
# Chai-1 (ESM mode — no MSAs needed)
emmentalembed fold --fasta proteins.fasta --method chai --output-dir pdb/

# Boltz-2 (open-source, MIT licensed)
emmentalembed fold --fasta proteins.fasta --method boltz --output-dir pdb/
```

## CLI Reference

```bash
emmentalembed embed     # Extract PLM embeddings
emmentalembed fold      # Predict structures (Chai-1 or Boltz-2)
emmentalembed version   # Show versions (package, torch, CUDA)
```

### Embedding extraction

```bash
emmentalembed embed \
    --fasta input.fasta \
    --model facebook/esm2_t33_650M_UR50D \
    --output embeddings.parquet \
    --batch-size 32 \
    --dtype float16 \
    --layer -1 \
    --representation mean
```

Or use a YAML config:
```bash
emmentalembed embed --config embed_config.yaml
```

### Structure prediction

```bash
# Chai-1 (ESM mode, no MSAs needed)
emmentalembed fold --fasta input.fasta --method chai --output-dir pdb/

# Boltz-2 (open-source, MIT licensed)
emmentalembed fold --fasta input.fasta --method boltz --output-dir pdb/
```

## Supported Models

Any HuggingFace-compatible protein language model works out of the box:

| Model | HuggingFace ID | Params | GPU |
|-------|---------------|--------|-----|
| ESM-2 650M | `facebook/esm2_t33_650M_UR50D` | 650M | A6000 |
| ESM-2 3B | `facebook/esm2_t36_3B_UR50D` | 3B | A6000 |
| ESM-2 15B | `facebook/esm2_t48_15B_UR50D` | 15B | A100 |
| ProtT5 XL | `Rostlab/prot_t5_xl_half_uniref50-enc` | 3B | A6000 |
| ANKH Large | `ElnaggarLab/ankh-large` | 1.5B | A6000 |
| AMPLIFY 350M | `chandar-lab/AMPLIFY_350M` | 350M | A6000 |
| ESM-C 300M | `esmc_300m` | 300M | A6000 |
| ProtBERT | `Rostlab/prot_bert` | ~420M | A6000 |

## SLURM Scripts

```bash
# Embedding extraction (default: ESM-2 650M on A6000)
sbatch scripts/embed.sh

# Override model/input
MODEL=facebook/esm2_t48_15B_UR50D FASTA=my_seqs.fasta sbatch scripts/embed.sh

# Chai-1 folding on A100
sbatch scripts/fold_chai.sh

# Boltz-2 folding on A100
sbatch scripts/fold_boltz.sh
```

## Environment Setup

```bash
conda create -n emmentalembed -c conda-forge python=3.12 uv pip -y
conda activate emmentalembed
uv pip install -e ".[dev]"   # includes embeddings, folding, and test deps
```

## YAML Config

```yaml
embed:
  fasta_path: proteins.fasta
  output_path: embeddings.parquet
  model: facebook/esm2_t33_650M_UR50D
  batch_size: 32
  dtype: float16

fold:
  fasta_path: proteins.fasta
  output_dir: pdb_output
  method: chai
  use_esm_embeddings: true
```

## Running Tests

```bash
conda activate emmentalembed
python -m pytest tests/ -v
```

## License

MIT License. See [LICENSE](LICENSE.txt) for details.
