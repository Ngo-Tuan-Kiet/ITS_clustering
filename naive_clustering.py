"""
This file implements a clustering algorithm based on graph isomorphism to group chemical reactions.

Glossary:
- Reaction Center: A graph representation of a chemical reaction.
- Reaction: A dictionary containing a Reaction Center and its calculated graph invariants.
- Cluster / Isomerism class: A set of reactions.
- Cluster Space: A set of clusters.
"""

import time
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict

from synutility.SynIO.data_type import load_from_pickle
from synutility.SynVis.graph_visualizer import GraphVisualizer
import networkx.algorithms.isomorphism as iso

# Type definitions
ReactionCenter = nx.Graph
Reaction = Dict[str, any]
Cluster = List[Reaction]
ClusterSpace = List[Cluster]

# Global variables
FILEPATH = 'ITS_graphs.pkl.gz'
FILEPATH_BIG = 'ITS_largerdataset.pkl.gz'
NB_RANGE = 1
BENCHMARK_CSV = f'benchmark_results_big_nb{NB_RANGE}.csv'


def load_reactions(filepath: str, nb_range: int = 0) -> List[Reaction]:
    """
    Loads a list of reactions from a pickle file.
    Returns the list of reactions.
    """
    print(f'Loading reactions from {filepath} with neighborhood range {nb_range}...')
    data = load_from_pickle(filepath)

    # Combine element and charge to create a unique identifier for each node (necessary for Weisfeiler-Lehman)
    for datum in data:
        graph = datum['ITS']
        for node in graph.nodes:
            graph.nodes[node]['elecharge'] = f'{graph.nodes[node]['element']}{graph.nodes[node]['charge']}'
    
    reactions = [{'graph': get_reaction_center(datum['ITS'], nb_range=nb_range)} for datum in data]
    print(f'Number of reactions loaded: {len(reactions)}')

    return reactions


def find_isomorphism_class(reaction: Reaction, cluster_space: ClusterSpace) -> int:
    """
    Attempts to find the isomorphism class of a reaction.
    Returns the index of the cluster if found, otherwise None.
    """
    for i, cluster in enumerate(cluster_space):
        if nx.is_isomorphic(
            cluster[0]['graph'], reaction['graph'],
            node_match=iso.categorical_node_match(["charge", "element"], [1, 'H']),
            edge_match=iso.categorical_edge_match("order", (1.0, 1.0))
        ):
            return i
        
    return None


# WP2: Clustering of reaction centers based on graph isomorphism
def naive_clustering(reactions: List[Reaction]) -> ClusterSpace:
    """
    Groups reactions into isomorphism classes.
    Returns the cluster space.
    """
    cluster_space = []

    for reaction in reactions:
        index = find_isomorphism_class(reaction, cluster_space)
        if index is not None:
            # Add to existing partition
            cluster_space[index].append(reaction)
        else:
            # Create a new partition
            cluster_space.append([reaction])

    return cluster_space


# WP3: Pre-filtering of reaction centers based on graph invariants
def get_graph_invariants(reaction: Reaction, relevant_invariants: List) -> Dict[str, any]:
    """
    Calculates all relevant graph invariants for a reaction center.
    Returns a dictionary of calculated invariants.
    """
    invariants = {}
    reaction_center = reaction['graph']
    for inv in relevant_invariants:
        match inv:
            case 'vertex_count':
                invariants[inv] = reaction_center.number_of_nodes()
            case 'edge_count':
                invariants[inv] = reaction_center.number_of_edges()
            case 'degree_sequence':
                invariants[inv] = sorted([d for n, d in reaction_center.degree()], reverse=True)
            case 'lex_node_sequence':
                invariants[inv] = sorted([reaction_center.nodes[n]['element'] 
                                          for n in reaction_center.nodes], reverse=True)
            # NetworkX function might be incorrect
            # case 'algebraic_connectivity':
            #     invariants[inv] = round(nx.linalg.algebraic_connectivity(reaction_center), 3)
            # FIXME: Correct the calculation of rank
            # case 'rank':
            #     invariants[inv] = reaction_center.number_of_nodes() - nx.number_connected_components(reaction_center)
            case 'wiener_index':
                invariants[inv] = nx.algorithms.wiener_index(reaction_center)
            # Estrada is not an invariant
            # case 'estrada_index':
            #     invariants[inv] = nx.algorithms.estrada_index(reaction_center)
            case 'wl1':
                invariants[inv] = nx.algorithms.weisfeiler_lehman_graph_hash(reaction_center, iterations=1, node_attr='elecharge', edge_attr='order')
            case 'wl2':
                invariants[inv] = nx.algorithms.weisfeiler_lehman_graph_hash(reaction_center, iterations=2, node_attr='elecharge', edge_attr='order')
            case 'wl3':
                invariants[inv] = nx.algorithms.weisfeiler_lehman_graph_hash(reaction_center, iterations=3, node_attr='elecharge', edge_attr='order')
            case _:
                print(f"Invalid invariant: {inv}, continuing...")

    return invariants


def compare_graph_invariants(reaction1: Dict[str, any], reaction2: Dict[str, any], relevant_invariants) -> bool:
    """
    Compares two reactions based on their graph invariants.
    Returns True if all invariants are equal, otherwise False.
    """
    reaction1_invariants = reaction1['invariants']
    reaction2_invariants = reaction2['invariants']
    
    for inv in relevant_invariants:
        if reaction1_invariants[inv] != reaction2_invariants[inv]:
            return False
        
    return True


def filter_by_invariants(reactions: List[Reaction], relevant_invariants: List) -> ClusterSpace:
    """
    Pre-filters reactions based on the invariants of their reaction centers.
    Returns the filtered cluster space.
    """
    cluster_space = []
    count = 0

    for current_reaction in reactions:
        count += 1
        current_reaction['invariants'] = get_graph_invariants(current_reaction, relevant_invariants)
        index = None
        for i, cluster in enumerate(cluster_space):
            if compare_graph_invariants(current_reaction, cluster[0], relevant_invariants):
                index = i
                break
        if index is not None:
            # Cluster found
            cluster_space[index].append(current_reaction)
        else:
            # Create a new cluster
            cluster_space.append([current_reaction])
    #         print(count)
    #         print(f"New partition added: {current_reaction['invariants']}")

    # print(f"Number of clusters after pre-filtering: {len(cluster_space)}")
    return cluster_space


def cluster_filtered_reactions(filtered_reactions: ClusterSpace) -> ClusterSpace:
    """
    Groups reactions into isomorphism classes after pre-filtering.
    Returns the cluster space.
    """
    final_cluster_space = []

    for group in filtered_reactions:
        final_clusters = naive_clustering(group)
        final_cluster_space.extend(final_clusters)

    return final_cluster_space


def get_reaction_center(
        its: nx.graph,
        element_key: list = ['element', 'charge', 'elecharge'],
        nb_range: int = 0
        ) -> ReactionCenter:
    """
    Finds the reaction center of an ITS graph. Neighborhood range and node keys can be specified.
    Returns the reaction center.
    """
    reaction_center_core = nx.Graph()

    for n1, n2, data in its.edges(data=True):
        if data['standard_order'] != 0:
            reaction_center_core.add_node(n1, **{k: its.nodes[n1][k] for k in element_key if k in its.nodes[n1]})
            reaction_center_core.add_node(n2, **{k: its.nodes[n2][k] for k in element_key if k in its.nodes[n2]})
            reaction_center_core.add_edge(n1, n2, **{k: data[k] for k in data})

    if nb_range <= 0:
        return reaction_center_core
    
    # Extend the reaction center to include the neighborhood of radius nb_range
    reaction_center = reaction_center_core.copy()
    for _ in range(nb_range):
        for node in reaction_center_core.nodes:
            for neighbor in its.neighbors(node):
                if neighbor not in reaction_center:
                    reaction_center.add_node(neighbor, **{k: its.nodes[neighbor][k] for k in element_key if k in its.nodes[neighbor]})
                reaction_center.add_edge(node, neighbor, **{k: its[node][neighbor][k] for k in its[node][neighbor]})
            reaction_center_core = reaction_center.copy()

    return reaction_center


def plot_partition(partition, partition_index=0):
    """Plots a specific partition."""
    fig, ax = plt.subplots(figsize=(15, 10))
    vis = GraphVisualizer()
    for rc in partition[partition_index]:
        vis.plot_its(rc, ax, use_edge_color=True)
        plt.show()
        fig, ax = plt.subplots(figsize=(15, 10))  # Create a new figure for the next plot


def plot_representatives(partition):
    """Plots a representative of each partition."""
    fig, ax = plt.subplots(figsize=(15, 10))
    vis = GraphVisualizer()
    for subset in partition:
        vis.plot_its(subset[0], ax, use_edge_color=True)
        plt.show()
        fig, ax = plt.subplots(figsize=(15, 10))  # Create a new figure for the next plot


def benchmark_clustering(invariant_list: List, reactions: List[Reaction], output_file: str = 'benchmark_results.csv') -> List[Dict]:
    """
    Tests a list of invariants and their performance.

    """
    results = []

    for invariant in invariant_list:
        print(f"Testing invariant: {invariant}")

        start_time_pre_filter = time.process_time()
        cluster_space_pre_filter = filter_by_invariants(reactions, invariant)
        end_time_pre_filter = time.process_time()
        pre_filter_time = end_time_pre_filter - start_time_pre_filter

        print('Pre-filtering done. Starting clustering...')
        start_time = time.process_time()
        cluster_space = cluster_filtered_reactions(cluster_space_pre_filter)
        end_time = time.process_time()
        cluster_time = end_time - start_time
        
        results.append({
            'Invariant': invariant,
            'CPU Time for pre-filtering (seconds)': pre_filter_time,
            'Number of Clusters after pre-filtering': len(cluster_space_pre_filter),
            'CPU Time for clustering (seconds)': cluster_time,
            'Number of Clusters': len(cluster_space),
            'Total CPU Time (seconds)': pre_filter_time + cluster_time
        })
    
    results_df = pd.DataFrame(results)
    print(f'Writing results to {output_file}...')
    results_df.to_csv(output_file, index=False)


def main():
    reactions = load_reactions(FILEPATH_BIG, nb_range=NB_RANGE)

    # print("Starting clustering...")
    # start_time = time.process_time()
    # # cluster_space = naive_clustering(reactions)
    # cluster_space = filter_by_invariants(reactions, ['wl1'])
    # cluster_space = cluster_filtered_reactions(cluster_space)
    # end_time = time.process_time()

    # print(f"Time taken: {end_time - start_time:.4f} seconds")
    # print(f"Total reaction centers: {len(reactions)}")
    # print(f"Number of clusters: {len(cluster_space)}")

    # Plot a specific partition (e.g., the fourth partition)
    # plot_partition(partition, partition_index=3)
    # plot_representatives(partition)

    # Benchmarking
    invariant_list = [
        ['vertex_count'],
        ['edge_count'],
        ['degree_sequence'],
        ['lex_node_sequence'],
        ['wiener_index'],
        ['wl1'],
        ['wl2'],
        ['wl3']
    ]

    benchmark_clustering(invariant_list, reactions, BENCHMARK_CSV)
    print("Done.")


if __name__ == "__main__":
    main()