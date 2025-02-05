import pandas as pd
import os
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def process_isoform_data(input_file, output_label_file, output_fasta_file, max_length=None, exclude_ids=None):
    """
    Process isoform data to create a label file and a FASTA file.
    
    Args:
        input_file (str): Path to input CSV file containing isoform data
        output_label_file (str): Path to save the label CSV file
        output_fasta_file (str): Path to save the FASTA file
        max_length (int, optional): Maximum sequence length to include. If None, include all sequences
        exclude_ids (list, optional): List of identifiers (Gene_Isoform) to exclude
    """
    # Make output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_fasta_file), exist_ok=True)
    os.makedirs(os.path.dirname(output_label_file), exist_ok=True)

    # Read the input CSV file
    df = pd.read_csv(input_file)
    
    # Create new identifier by combining Gene and Isoform with underscore
    df['identifier'] = df['Gene'] + '_' + df['Isoform']
    
    # Initialize list to store all excluded sequences and their reasons
    excluded_info = []
    
    # Filter by sequence length if max_length is specified
    if max_length is not None:
        df['seq_length'] = df['Sequence'].str.len()
        length_excluded_df = df[df['seq_length'] > max_length]
        
        # Store length-excluded sequences info
        for _, row in length_excluded_df.iterrows():
            excluded_info.append({
                'identifier': row['identifier'],
                'reason': f'Length exceeds {max_length} (actual: {row["seq_length"]})'
            })
        
        df = df[df['seq_length'] <= max_length]
        df = df.drop('seq_length', axis=1)
    
    # Filter by excluded identifiers if specified
    if exclude_ids is not None:
        id_excluded_df = df[df['identifier'].isin(exclude_ids)]
        
        # Store manually excluded sequences info
        for _, row in id_excluded_df.iterrows():
            excluded_info.append({
                'identifier': row['identifier'],
                'reason': 'Manually excluded'
            })
        
        df = df[~df['identifier'].isin(exclude_ids)]
    
    # Print information about excluded sequences
    if excluded_info:
        print(f"\nExcluded sequences ({len(excluded_info)} total):")
        for info in excluded_info:
            print(f"- {info['identifier']}: {info['reason']}")
        print()
    
    # Create label file (excluding sequence)
    label_df = df[['identifier', 'Gene', 'Isoform', 'Localization', 'Correct prediction?']]
    label_df.to_csv(output_label_file, index=False)
    
    # Create FASTA records
    records = []
    for _, row in df.iterrows():
        record = SeqRecord(
            Seq(row['Sequence']),
            id=row['identifier'],
            description=""
        )
        records.append(record)
    
    # Write FASTA file
    with open(output_fasta_file, 'w') as handle:
        SeqIO.write(records, handle, 'fasta')
    
    print(f"Created label file: {output_label_file}")
    print(f"Created FASTA file: {output_fasta_file}")
    print(f"Processed {len(df)} entries")