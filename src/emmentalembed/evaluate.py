import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_similarity
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

def calculate_similarities(embeddings_file, labels_file):
    """
    Calculate multiple similarity metrics between protein embeddings and merge with labels
    """
    # Read embeddings
    embeddings_df = pd.read_csv(embeddings_file, index_col=0)
    embeddings = embeddings_df.values
    protein_names = embeddings_df.index
    
    # Calculate different similarity/distance metrics
    similarities = {}
    
    # Cosine similarity
    cos_sim = sklearn_similarity(embeddings)
    similarities['cosine'] = pd.DataFrame(cos_sim, index=protein_names, columns=protein_names)
    
    # Euclidean distance
    euc_dist = squareform(pdist(embeddings, metric='euclidean'))
    similarities['euclidean'] = pd.DataFrame(euc_dist, index=protein_names, columns=protein_names)
    
    # Manhattan distance
    man_dist = squareform(pdist(embeddings, metric='cityblock'))
    similarities['manhattan'] = pd.DataFrame(man_dist, index=protein_names, columns=protein_names)
    
    # Pearson correlation
    pearson_corr = np.corrcoef(embeddings)
    similarities['pearson'] = pd.DataFrame(pearson_corr, index=protein_names, columns=protein_names)
    
    # Get unique base protein names (without the suffixes)
    base_proteins = set(name.split('_')[0] for name in protein_names)
    
    # Calculate pairwise similarities for each metric
    results_data = []
    
    for protein_base in base_proteins:
        # Get all variants for this protein
        variants = [name for name in protein_names if name.startswith(protein_base + '_')]
        
        # Skip if we don't have enough variants
        if len(variants) < 2:
            continue
            
        # Get the annotated variant
        annotated = next((v for v in variants if 'Annotated' in v), None)
        if not annotated:
            continue
            
        # Compare annotated with extended and truncated variants
        for variant in variants:
            if variant != annotated:  # Don't compare annotated with itself
                metric_values = {
                    f'{metric}_similarity': sim_df.loc[annotated, variant]
                    for metric, sim_df in similarities.items()
                }
                
                metric_values.update({
                    'Protein': protein_base,
                    'Variant_Type': 'Extended' if 'Extended' in variant else 'Truncated'
                })
                results_data.append(metric_values)
    
    similarities_df = pd.DataFrame(results_data)
    
    # Process labels data
    labels_df = pd.read_csv(labels_file, index_col=0)
    metadata = []
    
    for protein_base in base_proteins:
        # Get rows for this protein
        protein_rows = labels_df[labels_df['Gene'] == protein_base]
        if len(protein_rows) < 2:
            continue
            
        # Get annotated variant
        annotated_row = protein_rows[protein_rows.index.str.contains('Annotated')].iloc[0]
        
        # Compare with other variants
        other_rows = protein_rows[~protein_rows.index.str.contains('Annotated')]
        for _, variant_row in other_rows.iterrows():
            metadata.append({
                'Protein': protein_base,
                'Variant_Type': 'Extended' if 'Extended' in variant_row.name else 'Truncated',
                'Annotated_Localization': annotated_row['Localization'],
                'Variant_Localization': variant_row['Localization'],
                'Same_Localization': annotated_row['Localization'] == variant_row['Localization'],
                'Annotated_Correct': annotated_row['Correct prediction?'],
                'Variant_Correct': variant_row['Correct prediction?']
            })
    
    metadata_df = pd.DataFrame(metadata)
    
    # Merge similarities with metadata
    output_df = metadata_df.merge(similarities_df, on=['Protein', 'Variant_Type'])
    
    return output_df

def analyze_metric_correlations(df):
   """
   Analyze correlations between different similarity metrics and localization
   
   Parameters:
   -----------
   df : pandas.DataFrame
       DataFrame containing similarity scores and metadata
   
   Returns:
   --------
   dict
       Dictionary containing statistical measures for each metric
   """
   metrics = [col for col in df.columns if col.endswith('_similarity')]
   df['Same_Localization_Numeric'] = df['Same_Localization'].astype(int)
   
   results = {}
   
   for metric in metrics:
       # Calculate correlations and t-tests
       correlation = df['Same_Localization_Numeric'].corr(df[metric])
       same_loc = df[df['Same_Localization']][metric]
       diff_loc = df[~df['Same_Localization']][metric]
       t_stat, p_value = stats.ttest_ind(same_loc, diff_loc)
       
       results[metric] = {
           'correlation': correlation,
           'p_value': p_value,
           't_statistic': t_stat,
           'same_loc_mean': same_loc.mean(),
           'same_loc_std': same_loc.std(),
           'same_loc_count': len(same_loc),
           'diff_loc_mean': diff_loc.mean(),
           'diff_loc_std': diff_loc.std(),
           'diff_loc_count': len(diff_loc)
       }
       
       # Print results
       print(f"\nResults for {metric}:")
       print(f"Correlation with Same_Localization: {correlation:.3f}")
       print(f"T-test p-value: {p_value:.3f}")
       print("\nSummary Statistics:")
       print("Same Localization:")
       print(f"Mean: {results[metric]['same_loc_mean']:.3f}")
       print(f"Std: {results[metric]['same_loc_std']:.3f}")
       print(f"Count: {results[metric]['same_loc_count']}")
       print("\nDifferent Localization:")
       print(f"Mean: {results[metric]['diff_loc_mean']:.3f}")
       print(f"Std: {results[metric]['diff_loc_std']:.3f}")
       print(f"Count: {results[metric]['diff_loc_count']}")
     
   return results

def plot_protein_metrics(df, model_name):
    """
    Create a strip plot showing all similarity metrics for each protein using Seaborn
    """
    # Get metrics columns
    metrics = [col for col in df.columns if col.endswith('_similarity')]
    
    # Setup the plot
    fig, axs = plt.subplots(1, len(metrics), figsize=(22, 6))
    
    # Define interpretation for each metric
    metric_interpretation = {
        'cosine_similarity': 'Higher = More Similar',
        'euclidean_similarity': 'Lower = More Similar', 
        'manhattan_similarity': 'Lower = More Similar',
        'pearson_similarity': 'Higher = More Similar'
    }
    
    # Create separate subplot for each metric
    for ax_idx, metric in enumerate(metrics):
        ax = axs[ax_idx]
        
        # Create the strip plot and store the points
        scatter = sns.stripplot(
            data=df,
            y=metric,
            hue='Same_Localization',
            palette={True: 'blue', False: 'red'},
            size=8,
            alpha=0.6,
            jitter=0.4,
            ax=ax
        )
        
        # Get the plotted points
        points = scatter.collections[0].get_offsets().data
        
        # Create a mapping of y-values to x-coordinates
        y_to_x = {y: x for x, y in points}
        
        # Add protein labels using the actual point positions
        for idx, row in df.iterrows():
            y_val = row[metric]
            # Find the corresponding x position
            x_pos = y_to_x.get(y_val, 0)
            
            ax.annotate(
                row['Protein'],
                xy=(x_pos, y_val),  # Use the actual point position
                xytext=(5, 0),      # Small offset to the right
                textcoords='offset points',
                fontsize=8,
                alpha=0.8,
                horizontalalignment='left',
                verticalalignment='center'
            )
        
        # Customize each subplot
        metric_name = metric.replace('_similarity', '').capitalize()
        ax.set_title(f'{metric_name}\n({metric_interpretation[metric]})', 
                    fontsize=10, pad=10)
        
        # Remove x ticks and labels
        ax.set_xticks([])
        ax.set_xlabel('')
        
        # Only show y label for first plot
        if ax_idx > 0:
            ax.set_ylabel('')
            
        # Remove the legend for all but the last subplot
        if ax_idx < len(metrics) - 1:
            ax.get_legend().remove()
        else:
            ax.legend(title='Same localization?', bbox_to_anchor=(1.05, 0.5))
    
    # Add main title
    plt.suptitle(f'{model_name}\nProtein Similarity Metrics by Localization Status',
                fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.96, top=0.90, wspace=0.2)
    plt.show()
    
    return fig