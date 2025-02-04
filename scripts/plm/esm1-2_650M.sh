#!/bin/bash
# Configuration values for SLURM job submission.
# One leading hash ahead of the word SBATCH is not a comment, but two are.
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

cd /lab/barcheese01/mdiberna/plm_sandbox/

# esm 1
study_names=("isoform_sequences_esm1")

fasta_path="output/isoform/process/"
results_path="output/isoform/esm/"
models=("esm1b_t33_650M_UR50S" "esm1v_t33_650M_UR90S_1")

for model in "${models[@]}"; do
  model_names+=("sandbox/plm/esm/models/${model}.pt")
done

repr_layers=33
toks_per_batch=2000

mkdir -p ${results_path}

for model_name in "${model_names[@]}"; do
  for study in "${study_names[@]}"; do
    command="python3 sandbox/plm/esm/extract.py ${model_name} ${fasta_path}${study}.fasta ${results_path}${study}/${model_name} --toks_per_batch ${toks_per_batch} --include mean --concatenate_dir ${results_path}"
    echo "Running command: ${command}"
    eval "${command}"
  done
done

# esm 2
study_names=("isoform_sequences")
model_names=()

fasta_path="output/isoform/process/"
results_path="output/isoform/esm/"
models=("esm2_t33_650M_UR50D")

for model in "${models[@]}"; do
  model_names+=("sandbox/plm/esm/models/${model}.pt")
done

repr_layers=33
toks_per_batch=2000

mkdir -p ${results_path}

for model_name in "${model_names[@]}"; do
  for study in "${study_names[@]}"; do
    command="python3 sandbox/plm/esm/extract.py ${model_name} ${fasta_path}${study}.fasta ${results_path}${study}/${model_name} --toks_per_batch ${toks_per_batch} --include mean --concatenate_dir ${results_path}"
    echo "Running command: ${command}"
    eval "${command}"
  done
done