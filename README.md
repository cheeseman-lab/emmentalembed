# EmmentalEmbed

A toolkit for extracting embeddings from various protein language models (PLMs). This repository provides standardized interfaces for generating embeddings from protein sequences using different PLM architectures.

## Supported Models

- **ANKH**: Large and Base models
- **ESM**: 
  - ESM-2 (15B, 3B, 650M parameters)
  - ESM-1b (650M parameters)
  - ESM-1v (650M parameters)
- **ProtT5**: XL-U50 
- **ProteinBERT**: Base model
- **UniRep**: Original implementation
- **One-hot encoding**: Basic sequence encoding

## Project Structure

```
├── data/                     # Data directory
│   └── isoform/             # Isoform-specific data
├── sandbox/                  # Main package directory
│   ├── plm/                 # PLM implementations
│   │   ├── ankh/           # ANKH model
│   │   ├── esm/            # ESM models
│   │   ├── one-hot/        # One-hot encoding
│   │   ├── prot_t5/        # ProtT5 model
│   │   ├── proteinbert/    # ProteinBERT
│   │   └── unirep/         # UniRep model
│   └── src/                 # Core utilities
├── scripts/                  # Runtime scripts
│   ├── evaluate/            # Evaluation scripts
│   ├── plm/                 # SLURM job scripts
│   └── process/             # Data processing scripts
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/cheeseman-lab/plm_sandbox.git
cd plm_sandbox
```

2. Set up the embedding analysis environment:
```bash
conda env create -f environment.yml
conda activate emmentalembed
pip install -e .
```

3. Set up the PLM environment and download models:
```bash
conda env create -f plm_environment.yml
conda activate plm
./setup_plm.sh
```

## Usage

### Processing Data

Convert your protein sequences into the required format. For isoforms (example), we use the following approach:

```python
from sandbox.src.process import process_isoform_data

process_isoform_data(
    input_file='data/isoform/isoform_localization.csv',
    output_label_file='output/isoform/process/isoform_labels.csv',
    output_fasta_file='output/isoform/process/isoform_sequences.fasta'
)
```

You can add further functions to the process file for other types of proteins you'd like to process.

### Generating Embeddings

Each PLM has a standardized interface. Basic usage:

```bash
python sandbox/plm/<model>/extract.py -i input.fasta -o output.csv [additional_options]
```

Model-specific examples:

```bash
# ANKH
python sandbox/plm/ankh/extract.py -i input.fasta -o output.csv --model large

# ESM-2
python sandbox/plm/esm/extract.py esm2_t48_15B_UR50D input.fasta output_dir \
    --include mean --concatenate_dir results/

# ProtT5
python sandbox/plm/prot_t5/extract.py -i input.fasta -o output.csv --per_protein 1

# One-hot encoding
python sandbox/plm/one-hot/extract.py input.fasta --method one_hot --results_path results/
```

### SLURM Scripts

For HPC environments, use the provided SLURM scripts in `scripts/plm/`:

```bash
sbatch scripts/plm/<model>.sh
```

### Evaluating Embeddings

As an example, we evaluate the similarities between pairs of embeddings in `evaluate/evaluate_isoforms.ipynb`.

## Development

### Environment Management

The project uses two conda environments:

1. `emmentalembed`: For analysis and processing of embeddings
2. `plm`: For running protein language models and generating embeddings

### Package Structure

Main components:
- `sandbox.plm`: PLM implementations and extraction scripts
- `sandbox.src`: Core utilities for data processing
- `scripts`: Runtime and submission scripts
