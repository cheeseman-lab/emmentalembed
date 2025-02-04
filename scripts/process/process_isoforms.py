import os
from sandbox.src.process import process_isoform_data

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

process_isoform_data(
    input_file=os.path.join(project_root, 'data/isoform/isoform_localization.csv'),
    output_label_file=os.path.join(project_root, 'output/isoform/process/isoform_labels.csv'),
    output_fasta_file=os.path.join(project_root, 'output/isoform/process/isoform_sequences.fasta'))