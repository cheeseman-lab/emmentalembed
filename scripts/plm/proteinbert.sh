#!/bin/bash
# Configuration values for SLURM job submission.
# One leading hash ahead of the word SBATCH is not a comment, but two are.
#SBATCH --time=2:00:00 
#SBATCH --job-name=proteinbert
#SBATCH -n 1 
#SBATCH -N 1   
#SBATCH --partition=nvidia-2080ti-20
#SBATCH --gres=gpu:1                  
#SBATCH --cpus-per-task=1  
#SBATCH --mem=10gb  
#SBATCH --output out/proteinbert-%j.out 

source ~/.bashrc
conda activate plm

cd /lab/barcheese01/mdiberna/plm_sandbox/

study_names=("isoform_sequences")

fasta_path="output/isoform/process/"
results_path="output/isoform/proteinbert/"
model_names=("proteinbert")

mkdir -p ${results_path}

for model_name in "${model_names[@]}"; do
  for study in "${study_names[@]}"; do
    command="python sandbox/plm/proteinbert/extract.py --input ${fasta_path}${study}.fasta --output ${results_path}${study}_${model_name}.csv"
    echo "Running command: ${command}"
    eval "${command}"
  done
done