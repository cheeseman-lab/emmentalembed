#!/bin/bash
#SBATCH --time=2:00:00 
#SBATCH --job-name=unirep
#SBATCH -n 1 
#SBATCH -N 1   
#SBATCH --partition=nvidia-2080ti-20
#SBATCH --gres=gpu:1                  
#SBATCH --cpus-per-task=1  
#SBATCH --mem=10gb  
#SBATCH --output out/unirep-%j.out 

source ~/.bashrc
conda activate plm

# Get absolute path to project root
PROJECT_ROOT="/lab/barcheese01/mdiberna/emmentalembed"
PLM_DIR="${PROJECT_ROOT}/plm"
cd ${PLM_DIR}

study_names=("isoform_sequences")

fasta_path="${PROJECT_ROOT}/output/isoform/process"
results_path="${PROJECT_ROOT}/output/isoform/unirep"
model_names=("unirep")

# Create output directory with proper permissions
mkdir -p "${results_path}"

for model_name in "${model_names[@]}"; do
    for study in "${study_names[@]}"; do
        command="python src/unirep/extract.py --input ${fasta_path}/${study}.fasta --output ${results_path}/${study}_${model_name}.csv"
        echo "Running command: ${command}"
        eval "${command}"
    done
done