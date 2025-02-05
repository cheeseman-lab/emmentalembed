#!/bin/bash
#SBATCH --time=2:00:00 
#SBATCH --job-name=esm2_3B
#SBATCH -N 1   
#SBATCH --partition=nvidia-A100-20
#SBATCH --gres=gpu:1                  
#SBATCH --cpus-per-task=1  
#SBATCH --mem=50gb  
#SBATCH --output out/esm2_3B-%j.out 

source ~/.bashrc
conda activate plm

# Get absolute path to project root
PROJECT_ROOT="/lab/barcheese01/mdiberna/emmentalembed"  # Using the same root as first script
PLM_DIR="${PROJECT_ROOT}/plm"
cd ${PLM_DIR}

study_names=("isoform_sequences")

fasta_path="${PROJECT_ROOT}/output/isoform/process"
results_path="${PROJECT_ROOT}/output/isoform/esm"
models=("esm2_t36_3B_UR50D")

for model in "${models[@]}"; do
    model_names+=("${PLM_DIR}/models/${model}.pt")
done

repr_layers=36
toks_per_batch=256

# Create output directory with proper permissions
mkdir -p "${PROJECT_ROOT}/output/isoform/esm"

for model_name in "${model_names[@]}"; do
    for study in "${study_names[@]}"; do
        command="python src/esm/extract.py ${model_name} ${fasta_path}/${study}.fasta ${results_path}/${study}/${model_name##*/} --toks_per_batch ${toks_per_batch} --include mean --concatenate_dir ${results_path}"
        echo "Running command: ${command}"
        eval "${command}"
    done
done