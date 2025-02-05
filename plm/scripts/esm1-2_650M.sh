#!/bin/bash
#SBATCH --time=2:00:00 
#SBATCH --job-name=esm1-2_650M
#SBATCH -n 1 
#SBATCH -N 1   
#SBATCH --partition=nvidia-2080ti-20
#SBATCH --gres=gpu:1                  
#SBATCH --cpus-per-task=1  
#SBATCH --mem=10gb  
#SBATCH --output out/esm1-2_650M-%j.out 

source ~/.bashrc
conda activate plm

# Get absolute path to project root
PROJECT_ROOT="/lab/barcheese01/mdiberna/emmentalembed"  # Change this to the root of your project
PLM_DIR="${PROJECT_ROOT}/plm"

cd ${PLM_DIR}

# Create output directory with proper permissions
mkdir -p "${PROJECT_ROOT}/output/isoform/esm"

# ESM 1
study_names=("isoform_sequences_esm1")
fasta_path="${PROJECT_ROOT}/output/isoform/process"
results_path="${PROJECT_ROOT}/output/isoform/esm"
models=("esm1b_t33_650M_UR50S" "esm1v_t33_650M_UR90S_1")

model_names=()  # Initialize empty array
for model in "${models[@]}"; do
    model_names+=("${PLM_DIR}/models/${model}.pt")
done

repr_layers=33
toks_per_batch=2000

for model_name in "${model_names[@]}"; do
    for study in "${study_names[@]}"; do
        command="python src/esm/extract.py ${model_name} ${fasta_path}/${study}.fasta ${results_path}/${study}/${model_name##*/} --toks_per_batch ${toks_per_batch} --include mean --concatenate_dir ${results_path}"
        echo "Running command: ${command}"
        eval "${command}"
    done
done

# ESM 2
model_names=()  # Reset array
study_names=("isoform_sequences")
models=("esm2_t33_650M_UR50D")

for model in "${models[@]}"; do
    model_names+=("${PLM_DIR}/models/${model}.pt")
done

for model_name in "${model_names[@]}"; do
    for study in "${study_names[@]}"; do
        command="python src/esm/extract.py ${model_name} ${fasta_path}/${study}.fasta ${results_path}/${study}/${model_name##*/} --toks_per_batch ${toks_per_batch} --include mean --concatenate_dir ${results_path}"
        echo "Running command: ${command}"
        eval "${command}"
    done
done