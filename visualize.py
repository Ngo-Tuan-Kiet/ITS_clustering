"""
Handles the creation of all graphs needed for the presentation.
"""

import sys
import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = sys.argv[1]
DATA = pd.read_csv(DATA_PATH)
FIGSIZE = (20,10)

plt.rc('axes', titlesize=20)
plt.rc('axes', labelsize=16)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('legend', fontsize=16)


def plot_clustering_times(data: pd.DataFrame):
    """
    Plots clustering times in bar graph.
    """
    invariants = data['Invariant'].unique()
    filter_times = data['CPU Time for pre-filtering (seconds)']
    clustering_times = data['CPU Time for clustering (seconds)']

    # Plot the data, using stacked bars of different colors for filtering and clustering times
    fig, ax = plt.subplots(figsize=FIGSIZE, layout='tight')
    plt.xticks(rotation=45)
    ax.bar(invariants, filter_times, label='Pre-filtering time', color='blue')
    ax.bar(invariants, clustering_times, bottom=filter_times, label='Clustering time', color='lightblue')

    # Add labels and legend
    ax.set_xlabel('Invariants')
    ax.set_ylabel('CPU Time (seconds)')
    ax.set_title('CPU Time for pre-filtering and clustering')
    ax.legend()

    plt.show()


def main():
    plot_clustering_times(DATA)


if __name__ == '__main__':
    main()