#!/bin/bash
#SBATCH --job-name=${USER}_emmentalembed_af3
#SBATCH --partition=nvidia-A100-20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=scripts/out/%x_%j.out
#SBATCH --error=scripts/out/%x_%j.err

# EmmentalEmbed: AlphaFold3 structure prediction
#
# Prerequisites:
#   - AF3 weights must be downloaded (request: https://forms.gle/svvpY4u2jsHEwWYS6)
#   - Set WEIGHTS_PATH to your local copy
#
# Usage:
#   WEIGHTS_PATH=/path/to/af3.bin sbatch scripts/fold_af3.sh
#   FASTA=my_seqs.fasta WEIGHTS_PATH=/path/to/af3.bin sbatch scripts/fold_af3.sh

set -euo pipefail

# Use project-local caches (avoid filling home directory)
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"

FASTA="${FASTA:-input.fasta}"
OUTPUT_DIR="${OUTPUT_DIR:-pdb_output_af3}"
WEIGHTS_PATH="${WEIGHTS_PATH:-weights/af3.bin}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
NUM_RECYCLES="${NUM_RECYCLES:-3}"

echo "=== EmmentalEmbed AlphaFold3 Folding ==="
echo "Input:       ${FASTA}"
echo "Output:      ${OUTPUT_DIR}"
echo "Weights:     ${WEIGHTS_PATH}"
echo "Max seq len: ${MAX_SEQ_LEN}"
echo "Recycles:    ${NUM_RECYCLES}"
echo "GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

eval "$(conda shell.bash hook)"
conda activate emmentalembed

mkdir -p "${OUTPUT_DIR}"
mkdir -p scripts/out

emmentalembed fold \
    --fasta "${FASTA}" \
    --method af3 \
    --weights "${WEIGHTS_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-seq-len "${MAX_SEQ_LEN}" \
    --num-recycles "${NUM_RECYCLES}"

echo ""
echo "=== Done ==="
