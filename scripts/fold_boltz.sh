#!/bin/bash
#SBATCH --job-name=${USER}_emmentalembed_boltz
#SBATCH --partition=nvidia-A100-20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=scripts/out/%x_%j.out
#SBATCH --error=scripts/out/%x_%j.err

# EmmentalEmbed: Boltz-2 structure prediction
#
# Usage:
#   sbatch scripts/fold_boltz.sh
#   FASTA=my_seqs.fasta OUTPUT_DIR=my_pdbs sbatch scripts/fold_boltz.sh

set -euo pipefail

# Use project-local caches (avoid filling home directory)
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"

FASTA="${FASTA:-input.fasta}"
OUTPUT_DIR="${OUTPUT_DIR:-pdb_output_boltz}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
NUM_RECYCLES="${NUM_RECYCLES:-3}"

echo "=== EmmentalEmbed Boltz-2 Folding ==="
echo "Input:       ${FASTA}"
echo "Output:      ${OUTPUT_DIR}"
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
    --method boltz \
    --output-dir "${OUTPUT_DIR}" \
    --max-seq-len "${MAX_SEQ_LEN}" \
    --num-recycles "${NUM_RECYCLES}"

echo ""
echo "=== Done ==="
