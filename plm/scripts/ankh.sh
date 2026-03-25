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

# Get absolute path to project root
PROJECT_ROOT="/lab/barcheese01/mdiberna/emmentalembed"  # Change this to the root of your project
PLM_DIR="${PROJECT_ROOT}/plm"
cd ${PLM_DIR}

# Add huggingface cache to your preferred path
export HF_HOME="${PLM_DIR}/src/ankh/huggingface_cache"
mkdir -p $HF_HOME

source ~/.bashrc
conda activate plm

# Define variables
study_names=("isoform_sequences")

fasta_path="${PROJECT_ROOT}/output/isoform/process"
results_path="${PROJECT_ROOT}/output/isoform/ankh"
model_names=("base" "large")

# Create output directory with proper permissions
mkdir -p "${results_path}"

for model_name in "${model_names[@]}"; do
  for study in "${study_names[@]}"; do
    command="python src/ankh/extract.py --input ${fasta_path}/${study}.fasta --output ${results_path}/${study}_ankh_${model_name}.csv --model ${model_name}"
    echo "Running command: ${command}"
    eval "${command}"
  done
done