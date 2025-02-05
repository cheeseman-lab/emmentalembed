#!/bin/bash
#SBATCH --time=2:00:00 
#SBATCH --job-name=ankh
#SBATCH -n 1 
#SBATCH -N 1   
#SBATCH --partition=nvidia-2080ti-20
#SBATCH --gres=gpu:1                  
#SBATCH --cpus-per-task=1  
#SBATCH --mem=10gb  
#SBATCH --output out/ankh-%j.out 

source ~/.bashrc
conda activate plm

# Get absolute path to project root
PROJECT_ROOT="/lab/barcheese01/mdiberna/emmentalembed"  # Change this to the root of your project
PLM_DIR="${PROJECT_ROOT}/plm"

# Create output directory with proper permissions
mkdir -p "${PROJECT_ROOT}/output/isoform/ankh"

study_names=("isoform_sequences")
fasta_path="${PROJECT_ROOT}/output/isoform/process"
results_path="${PROJECT_ROOT}/output/isoform/ankh"
model_names=("base" "large")

cd ${PLM_DIR}

for model_name in "${model_names[@]}"; do
  for study in "${study_names[@]}"; do
    command="python src/ankh/extract.py --input ${fasta_path}/${study}.fasta --output ${results_path}/${study}_ankh_${model_name}.csv --model ${model_name}"
    echo "Running command: ${command}"
    eval "${command}"
  done
done