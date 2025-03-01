import os
from emmentalembed.process import process_isoform_data

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

process_isoform_data(
    input_file=os.path.join(project_root, 'data/isoform/isoform_localization.csv'),
    output_label_file=os.path.join(project_root, 'output/isoform/process/isoform_labels.csv'),
    output_fasta_file=os.path.join(project_root, 'output/isoform/process/isoform_sequences.fasta')
)

process_isoform_data(
    input_file=os.path.join(project_root, 'data/isoform/isoform_localization.csv'),
    output_label_file=os.path.join(project_root, 'output/isoform/process/isoform_labels.csv'),
    output_fasta_file=os.path.join(project_root, 'output/isoform/process/isoform_sequences_esm1.fasta'),
    max_length=1024,
    exclude_ids=['TOP3A_Annotated']
)