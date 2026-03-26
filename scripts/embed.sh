#!/bin/bash
#SBATCH --job-name=${USER}_emmentalembed_embed
#SBATCH --partition=nvidia-A6000-20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=scripts/out/%x_%j.out
#SBATCH --error=scripts/out/%x_%j.err

# EmmentalEmbed: PLM embedding extraction
#
# Usage:
#   sbatch scripts/embed.sh                          # defaults (ESM2 650M)
#   MODEL=facebook/esm2_t36_3B_UR50D sbatch scripts/embed.sh
#   FASTA=my_seqs.fasta OUTPUT=my_emb.csv sbatch scripts/embed.sh
#
# For large models (ESM2 15B), use A100:
#   PARTITION=nvidia-A100-20 MODEL=facebook/esm2_t48_15B_UR50D MEM=100G sbatch scripts/embed.sh

set -euo pipefail

# Use project-local caches (avoid filling home directory)
# SLURM_SUBMIT_DIR is the directory from which sbatch was called;
# $0 is unreliable inside SLURM (points to spool copy).
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"

# Configurable parameters (override via environment)
FASTA="${FASTA:-input.fasta}"
OUTPUT="${OUTPUT:-embeddings.parquet}"
MODEL="${MODEL:-facebook/esm2_t33_650M_UR50D}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DTYPE="${DTYPE:-float16}"
LAYER="${LAYER:--1}"
REPRESENTATION="${REPRESENTATION:-mean}"

# Allow partition/resource override for large models
if [ -n "${PARTITION:-}" ]; then
    echo "Note: PARTITION override requested but must be set in #SBATCH directives."
    echo "For A100, copy this script and change the #SBATCH --partition line."
fi

echo "=== EmmentalEmbed Embedding Extraction ==="
echo "Model:  ${MODEL}"
echo "Input:  ${FASTA}"
echo "Output: ${OUTPUT}"
echo "Batch:  ${BATCH_SIZE}"
echo "dtype:  ${DTYPE}"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# Activate environment
eval "$(conda shell.bash hook)"
conda activate emmentalembed

# Create output directory
mkdir -p "$(dirname "${OUTPUT}")"
mkdir -p scripts/out

# Run extraction
emmentalembed embed \
    --fasta "${FASTA}" \
    --model "${MODEL}" \
    --output "${OUTPUT}" \
    --batch-size "${BATCH_SIZE}" \
    --dtype "${DTYPE}" \
    --layer "${LAYER}" \
    --representation "${REPRESENTATION}"

echo ""
echo "=== Done ==="
