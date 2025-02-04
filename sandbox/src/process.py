import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def process_isoform_data(input_file, output_label_file, output_fasta_file):
    """
    Process isoform data to create a label file and a FASTA file.
    
    Args:
        input_file (str): Path to input CSV file containing isoform data
        output_label_file (str): Path to save the label CSV file
        output_fasta_file (str): Path to save the FASTA file
    """
    # Read the input CSV file
    df = pd.read_csv(input_file)
    
    # Create new identifier by combining Gene and Isoform with underscore
    df['identifier'] = df['Gene'] + '_' + df['Isoform']
    
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