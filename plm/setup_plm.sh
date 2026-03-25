#!/bin/bash

echo "Setting up complete PLM environment..."

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create necessary directories
mkdir -p external

# Check if conda is initialized in bash
if ! command -v conda &> /dev/null; then
    echo "Conda not found in path. Please make sure conda is installed and initialized."
    echo "You might need to run: source ~/.bashrc"
    exit 1
fi

# Remove existing environment if it exists
conda remove -n plm --all -y

# Create new environment
echo "Creating new conda environment 'plm'..."
conda create -n plm python=3.10 -y

# Activate the environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate plm

if [ $? -ne 0 ]; then
    echo "Failed to activate conda environment"
    exit 1
fi

# Add channels in correct order
conda config --env --add channels defaults
conda config --env --add channels pytorch
conda config --env --add channels conda-forge

# Install conda packages
conda install -y \
    "numpy>=1.22,<=1.24.3" \
    cudatoolkit \
    "pytorch<2.0.0" \
    pandas \
    biopython \
    pip \
    gcc


# Install pip packages
pip install \
    ankh \
    fair-esm \
    jax-unirep \
    "transformers>=4.20.0" \
    "tensorflow==2.15.0" \
    pathlib
    
# Clone and setup ProteinBERT
if [ ! -d "external/proteinbert" ]; then
    echo "Setting up ProteinBERT..."
    git clone https://github.com/nadavbra/protein_bert.git external/proteinbert
    cd external/proteinbert
    git submodule init
    git submodule update
    python setup.py install
    cd "${SCRIPT_DIR}"
else
    echo "ProteinBERT already installed"
    cd external/proteinbert
    git pull
    python setup.py install
    cd "${SCRIPT_DIR}"
fi

# Revert to pandas 1.5.3    
pip install pandas==1.5.3

# Download the ESM models
echo "Downloading ESM models..."
python src/esm/download_models.py

echo "
=== Installation Verification ===
"

echo "Checking package versions..."
pip list | grep -E "tensorflow|keras|cudnn|torch|transformers|numpy"

echo "
=== Checking GPU Status ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "nvidia-smi not found. GPU might not be available."
fi

echo "Setup script completed successfully!"