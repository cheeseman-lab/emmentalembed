#!/bin/bash
#SBATCH --job-name=${USER}_emmentalembed_chai
#SBATCH --partition=nvidia-A100-20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=scripts/out/%x_%j.out
#SBATCH --error=scripts/out/%x_%j.err

# EmmentalEmbed: Chai-1 structure prediction
#
# Usage:
#   sbatch scripts/fold_chai.sh
#   FASTA=my_seqs.fasta OUTPUT_DIR=my_pdbs sbatch scripts/fold_chai.sh

set -euo pipefail

# Use project-local caches (avoid filling home directory)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"

FASTA="${FASTA:-input.fasta}"
OUTPUT_DIR="${OUTPUT_DIR:-pdb_output}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
NUM_RECYCLES="${NUM_RECYCLES:-3}"

echo "=== EmmentalEmbed Chai-1 Folding ==="
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
    --method chai \
    --output-dir "${OUTPUT_DIR}" \
    --max-seq-len "${MAX_SEQ_LEN}" \
    --num-recycles "${NUM_RECYCLES}" \
    --esm

echo ""
echo "=== Done ==="
