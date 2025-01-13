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

def plot_pca(principal_components, data, metadata=None, group=None, shape=None):
    plt.figure(figsize=(8, 6))
    
    merged_data = pd.DataFrame(principal_components, index=data.index)
    
    if metadata is not None:
        if group:
            merged_data = merged_data.merge(metadata[[group]], left_index=True, right_index=True)
        if shape:
            merged_data = merged_data.merge(metadata[[shape]], left_index=True, right_index=True)
    
    colors = merged_data[group].unique() if group else [None]
    shapes = merged_data[shape].unique() if shape else [None]
    
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'x', 'd', '|', '_']
    color_map = plt.get_cmap('tab10')
    
    for i, color in enumerate(colors):
        for j, shape_marker in enumerate(shapes):
            idx = (merged_data[group] == color) if group else True
            idx = idx & (merged_data[shape] == shape_marker) if shape else idx
            plt.scatter(merged_data.loc[idx, 0], merged_data.loc[idx, 1], s=50, marker=markers[j], color=color_map(i))
    
    # Create a legend for colors
    if group:
        for i, color in enumerate(colors):
            plt.scatter([], [], color=color_map(i), label=color)
    
    # Create a legend for shapes
    if shape:
        for shape_marker, marker in zip(shapes, markers):
            plt.scatter([], [], c='k', marker=marker, label=shape_marker)

    if group or shape:
        if group and shape:
            title = f"{group} (group) and {shape} (shape)"
        elif group:
            title = f"{group} (group)"
        else:
            title = f"{shape} (shape)"
        plt.legend(title=title, bbox_to_anchor=(1.05, 1), loc='upper left')
    
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
@click.option('--shape', type=str, help='Column name in metadata file to shape by')
def main(file_path, transpose, metadata, group, shape):
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
    plot_pca(principal_components, data, metadata_df, group, shape)
    
    # Print explained variance ratio
    print("Explained variance ratio:", pca.explained_variance_ratio_)

if __name__ == "__main__":
    main()