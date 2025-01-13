import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import click

def load_data(file_path):
    # Determine the file extension
    _, file_extension = os.path.splitext(file_path)
    
    # Read the file based on its extension
    if file_extension == '.csv':
        data = pd.read_csv(file_path, index_col=0)
    elif file_extension == '.tsv':
        data = pd.read_csv(file_path, sep='\t', index_col=0)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or TSV file.")
    
    return data

def perform_pca(data, n_components=2):
    # Standardize the data
    features = data.columns
    x = data.loc[:, features].values
    x_standardized = StandardScaler().fit_transform(x)
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(x_standardized)
    
    return principal_components, pca

def plot_pca(principal_components, data, metadata=None, group=None):
    plt.figure(figsize=(8, 6))
    
    if metadata is not None and group is not None:
        merged_data = pd.DataFrame(principal_components, index=data.index)
        merged_data = merged_data.merge(metadata[[group]], left_index=True, right_index=True)
        groups = merged_data[group].unique()
        
        for g in groups:
            idx = merged_data[group] == g
            plt.scatter(merged_data.loc[idx, 0], merged_data.loc[idx, 1], label=g, s=50)
        plt.legend()
    else:
        plt.scatter(principal_components[:, 0], principal_components[:, 1], c='blue', s=50)
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('2 Component PCA')
    plt.grid()
    plt.show()

@click.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--transpose', is_flag=True, help='Transpose the input data before applying PCA')
@click.option('--metadata', type=click.Path(exists=True), help='Path to the metadata file')
@click.option('--group', type=str, help='Column name in metadata file to group by')
def main(file_path, transpose, metadata, group):
    # Load data
    data = load_data(file_path)
    
    # Transpose data if --transpose flag is set
    if transpose:
        data = data.transpose()
    
    print(data)
    
    # Load metadata if provided
    metadata_df = None
    if metadata:
        metadata_df = load_data(metadata)
    
    # Perform PCA
    principal_components, pca = perform_pca(data)
    
    # Plot PCA results
    plot_pca(principal_components, data, metadata_df, group)
    
    # Print explained variance ratio
    print("Explained variance ratio:", pca.explained_variance_ratio_)

if __name__ == "__main__":
    main()