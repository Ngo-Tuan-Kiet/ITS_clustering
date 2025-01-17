"""
Handles the creation of all graphs needed for the presentation.
"""

import sys
import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = sys.argv[1]
DATA = pd.read_csv(DATA_PATH)
FIGSIZE = (24,12)

plt.rc('axes', titlesize=32)
plt.rc('axes', labelsize=24)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=16)


def plot_clustering_times(data: pd.DataFrame):
    """
    Plots clustering times in bar graph.
    """
    invariants = data['Invariant'].unique()
    filter_times = data['CPU Time for pre-filtering (seconds)']
    # Convert seconds to minutes
    filter_times = [time / 60 for time in filter_times]
    found_clusters = data['Number of Clusters after pre-filtering']
    clustering_times = data['CPU Time for clustering (seconds)']
    # Convert seconds to minutes
    clustering_times = [time / 60 for time in clustering_times]

    # Plot the data, using stacked bars of different colors for filtering and clustering times
    fig, ax = plt.subplots(figsize=FIGSIZE, layout='tight')
    plt.xticks(rotation=75)
    ax.bar(invariants, filter_times, label='Pre-filtering time', color='blue')
    ax.bar(invariants, clustering_times, bottom=filter_times, label='Clustering time', color='lightblue')
    # Add cluster number above each bar
    for i, cluster_num in enumerate(found_clusters):
        ax.text(i, filter_times[i] + clustering_times[i], f'{cluster_num}', ha='center', va='bottom', fontsize=16)

    # Add labels and legend
    ax.set_xlabel('Invariants')
    ax.set_ylabel('CPU Time (seconds)')
    ax.set_title('CPU Times and Number of Clusters for each Invariant (Neighborhood Size = 3)')
    ax.legend()

    plt.savefig('clustering_times.png')


def main():
    plot_clustering_times(DATA)


if __name__ == '__main__':
    main()
