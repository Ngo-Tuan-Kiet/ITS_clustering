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
from typing import List, Dict

from synutility.SynIO.data_type import load_from_pickle
from synutility.SynVis.graph_visualizer import GraphVisualizer
import networkx.algorithms.isomorphism as iso


# Type definitions
ReactionCenter = nx.Graph
Reaction = Dict[str, any]
Cluster = List[Reaction]
ClusterSpace = List[Cluster]


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
            case 'algebraic_connectivity':
                invariants[inv] = round(nx.linalg.algebraic_connectivity(reaction_center), 3)
            # TODO: Fix rank calculation
            # case 'rank':
            #     invariants[inv] = reaction_center.number_of_nodes() - nx.number_connected_components(reaction_center)
            case 'wl1':
                invariants[inv] = nx.algorithms.weisfeiler_lehman_graph_hash(reaction_center, iterations=1, node_attr='elecharge', edge_attr='order')
            case 'wl2':
                invariants[inv] = nx.algorithms.weisfeiler_lehman_graph_hash(reaction_center, iterations=2, node_attr='elecharge', edge_attr='order')
            case 'wl3':
                invariants[inv] = nx.algorithms.weisfeiler_lehman_graph_hash(reaction_center, iterations=3, node_attr='elecharge', edge_attr='order')

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
            print(count)
            print(f"New partition added: {current_reaction['invariants']}")

    print(f"Number of clusters after pre-filtering: {len(cluster_space)}")
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
        bond_key: str = 'order',
        standard_order: str = 'standard_order'
        ) -> ReactionCenter:
    """
    Finds the reaction center of an ITS graph using standard order edge attributes.
    Returns the reaction center.
    """
    reaction_center = nx.Graph()

    for n1, n2, data in its.edges(data=True):
        if data['standard_order'] != 0:
            reaction_center.add_node(n1, **{k: its.nodes[n1][k] for k in element_key if k in its.nodes[n1]})
            reaction_center.add_node(n2, **{k: its.nodes[n2][k] for k in element_key if k in its.nodes[n2]})
            reaction_center.add_edge(n1, n2, **{k: data[k] for k in data})

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


def main():
    filepath = 'ITS_graphs.pkl.gz'
    data = load_from_pickle(filepath)

    # Combine element and charge to create a unique identifier for each node (necessary for Weisfeiler-Lehman)
    for datum in data:
        graph = datum['ITS']
        for node in graph.nodes:
            graph.nodes[node]['elecharge'] = f'{graph.nodes[node]['element']}{graph.nodes[node]['charge']}'
    
    reactions = [{'graph': get_reaction_center(datum['ITS'])} for datum in data]

    print("Starting clustering...")
    start_time = time.process_time()
    # cluster_space = naive_clustering(reactions)
    cluster_space = filter_by_invariants(reactions, ['wl1'])
    cluster_space = cluster_filtered_reactions(cluster_space)
    end_time = time.process_time()

    print(f"Time taken: {end_time - start_time:.4f} seconds")
    print(f"Total reaction centers: {len(reactions)}")
    print(f"Number of clusters: {len(cluster_space)}")
    # Plot a specific partition (e.g., the fourth partition)
    # plot_partition(partition, partition_index=3)
    # plot_representatives(partition)


if __name__ == "__main__":
    main()