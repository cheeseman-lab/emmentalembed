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
emmentalembed/
├── src/
│   └── emmentalembed/         # Main package
│       ├── __init__.py
│       ├── evaluate.py        # Evaluation utilities
│       └── process.py         # Data processing
├── plm/                       # PLM subpackage
│   ├── src/
│   │   └── plm/
│   │       ├── ankh/         # ANKH model
│   │       ├── esm/          # ESM models
│   │       ├── one_hot/      # One-hot encoding
│   │       ├── prot_t5/      # ProtT5 model
│   │       ├── proteinbert/  # ProteinBERT
│   │       └── unirep/       # UniRep model
│   ├── scripts/              # PLM-specific scripts
│   │   └── extract/          # Extraction scripts
│   ├── pyproject.toml        # PLM dependencies
│   └── setup_plm.sh         # PLM setup script
├── scripts/                   # General scripts
│   ├── process/
│   └── evaluate/
└── pyproject.toml            # Main package dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/cheeseman-lab/emmentalembed.git
cd emmentalembed
```

2. Set up the main environment for embedding analysis:
```bash
# Create and activate main environment
conda create -n emmentalembed python=3.10
conda activate emmentalembed

# Install main package in development mode
pip install -e .
```

3. Set up the PLM environment (in a separate terminal):
```bash
# Create and activate PLM environment
conda create -n plm python=3.10
conda activate plm

# Install PLM package and dependencies
cd plm
pip install -e .

# Download and set up models
./setup_plm.sh
```

## Usage

### Processing Sequences

Convert your protein sequences to the required format:

```python
from emmentalembed.process import process_isoform_data

process_isoform_data(
    input_file='data/examples/isoforms.csv',
    output_label_file='results/labels.csv',
    output_fasta_file='results/sequences.fasta'
)
```

### Generating Embeddings

Each PLM has a standardized interface:

```bash
# Activate PLM environment first
conda activate plm

# Basic usage
python -m plm.<model>.extract -i input.fasta -o output.csv [options]

# Model-specific examples:

# ANKH
python -m plm.ankh.extract -i input.fasta -o output.csv --model large

# ESM-2
python -m plm.esm.extract esm2_t48_15B_UR50D input.fasta output_dir \
    --include mean --concatenate_dir results/

# ProtT5
python -m plm.prot_t5.extract -i input.fasta -o output.csv --per_protein 1

# One-hot encoding
python -m plm.one_hot.extract input.fasta --method one_hot --results_path results/
```

### SLURM Integration

For HPC environments, use the provided SLURM scripts:

```bash
# Submit specific model job
sbatch plm/scripts/extract/<model>.sh

# Or submit all models
for script in plm/scripts/extract/*.sh; do
    sbatch $script
done
```

### Analyzing Embeddings

```python
# Activate main environment
conda activate emmentalembed

# Import analysis tools
from emmentalembed.evaluate import compare_embeddings

# Load and analyze embeddings
results = compare_embeddings(
    'results/embeddings/*.csv',
    labels='results/labels.csv'
)
```

## Development

The project uses two separate environments to keep dependencies clean:

1. `emmentalembed`: For embedding analysis and processing
   - Lighter dependencies (pandas, numpy, scikit-learn)
   - Used for data processing and evaluation

2. `plm`: For running protein language models
   - Heavier dependencies (torch, tensorflow)
   - Used for generating embeddings from sequences

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

## License

MIT License. See [LICENSE](LICENSE.txt) for details.