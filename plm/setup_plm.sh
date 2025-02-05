#!/bin/bash

echo "Setting up PLM environment..."

# Ensure we're in the PLM directory
cd "$(dirname "$0")"

# Create necessary directories
mkdir -p external models

# Install pip packages
echo "Installing required packages..."
pip install ankh fair-esm jax-unirep "transformers>=4.20.0" torch==2.5.0 "tensorflow>=2.0" tf-keras

# Clone and setup ProteinBERT
if [ ! -d "external/proteinbert" ]; then
    echo "Setting up ProteinBERT..."
    git clone https://github.com/nadavbra/protein_bert.git external/proteinbert
    cd external/proteinbert
    git submodule init
    git submodule update
    python setup.py install
    cd ../..
else
    echo "ProteinBERT already installed"
fi

# Download the ESM models
echo "Downloading ESM models..."
python src/esm/download_models.py

echo "PLM setup complete!"