#!/bin/bash
# Configuration values for SLURM job submission.
# One leading hash ahead of the word SBATCH is not a comment, but two are.
#SBATCH --time=2:00:00 
#SBATCH --job-name=one-hot
#SBATCH -n 1 
#SBATCH -N 1   
#SBATCH --partition=20
#SBATCH --cpus-per-task=12
#SBATCH --mem=100gb  
#SBATCH --output out/one-hot-%j.out 

# SKIP DUE TO DIFFERENT SEQUENCE LENGTHS

source ~/.bashrc
conda activate plm

cd /lab/barcheese01/mdiberna/plm_sandbox/

study_names=("isoform_sequences")

fasta_path="output/isoform/process/"
results_path="output/isoform/one-hot/"
encoding_methods=("one_hot" "integer")

mkdir -p ${results_path}

for method in "${encoding_methods[@]}"; do
  for study in "${study_names[@]}"; do
    command="python3 sandbox/plm/one-hot/extract.py ${fasta_path}${study}.fasta --method ${method} --results_path ${results_path}"
    echo "Running command: ${command}"
    eval "${command}"
  done
done