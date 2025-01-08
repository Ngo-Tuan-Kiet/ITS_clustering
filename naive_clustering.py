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
import pandas as pd

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
            case 'rank':
                invariants[inv] = nx.linalg.graphmatrix.adjacency_matrix(reaction_center).todense().rank()
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
        standard_order: str = 'standard_order',
        nb_range: int = 0
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
    
    rc_nodes = set(reaction_center.nodes)
    rc_with_neighbors = nx.Graph()

    for n in rc_nodes:
        neighbor_graph = nx.ego_graph(its, n, radius=nb_range)

        rc_with_neighbors = nx.compose(rc_with_neighbors, neighbor_graph)
        
    rc_with_neighbors_subgraph = its.subgraph(rc_with_neighbors).copy()

    return rc_with_neighbors_subgraph


def plot_partition(partition, partition_index=0):
    """Plots a specific partition."""
    fig, ax = plt.subplots(figsize=(15, 10))
    vis = GraphVisualizer()
    for rc in partition[partition_index]:
        vis.plot_its(rc['graph'], ax, use_edge_color=True)
        plt.show()
        fig, ax = plt.subplots(figsize=(15, 10))  # Create a new figure for the next plot


def plot_representatives(partition):
    """Plots a representative of each partition."""
    fig, ax = plt.subplots(figsize=(15, 10))
    vis = GraphVisualizer()
    for subset in partition:
        vis.plot_its(subset[0]['graph'], ax, use_edge_color=True)
        plt.show()
        fig, ax = plt.subplots(figsize=(15, 10))  # Create a new figure for the next plot

def measure_clustering_time(reactions, invariants_list):
    results = []

    for invariants in invariants_list:
        start_time_pre_filter = time.process_time()
        cluster_space_pre_filter = filter_by_invariants(reactions, invariants)
        end_time_pre_filter = time.process_time()
        start_time = time.process_time()
        cluster_space = cluster_filtered_reactions(cluster_space_pre_filter)
        end_time = time.process_time()
        
        cpu_time = end_time - start_time
        results.append({
            'Invariants': invariants,
            'CPU Time for pre-filtering (seconds)': end_time_pre_filter - start_time_pre_filter,
            'Number of Clusters after pre-filtering': len(cluster_space_pre_filter),
            'CPU Time for clustering (seconds)': cpu_time,
            'Number of Clusters': len(cluster_space)
        })
    
    return results


# print("Loading small data...")
# filepath = 'ITS_graphs.pkl.gz'
# data = load_from_pickle(filepath)

# # Combine element and charge to create a unique identifier for each node (necessary for Weisfeiler-Lehman)
# for datum in data:
#     graph = datum['ITS']
#     for node in graph.nodes:
#         graph.nodes[node]['elecharge'] = f'{graph.nodes[node]["element"]}{graph.nodes[node]["charge"]}'

# reactions = [{'R-id': datum['R-id'],
#               'graph': get_reaction_center(datum['ITS'], nb_range=0)} for datum in data]
# reactions_with_neighbors = [{'R-id': datum['R-id'],
#                              'graph': get_reaction_center(datum['ITS'], nb_range=1)} for datum in data]


print("Loading large data...")
filepath_big = 'ITS_largerdataset.pkl.gz'
data_big = load_from_pickle(filepath_big)

# Combine element and charge to create a unique identifier for each node (necessary for Weisfeiler-Lehman)
for datum in data_big:
    graph = datum['ITS']
    for node in graph.nodes:
        graph.nodes[node]['elecharge'] = f'{graph.nodes[node]["element"]}{graph.nodes[node]["charge"]}'

print("Reactions big...")
reactions_big = [{'R_ID': datum['R_ID'],
              'graph': get_reaction_center(datum['ITS'], nb_range=0)} for datum in data_big]
print("Reactions big done")
reactions_big_with_neighbors = [{'R_ID': datum['R_ID'],
                             'graph': get_reaction_center(datum['ITS'], nb_range=1)} for datum in data_big]




print("Starting clustering...")
start_time = time.process_time()
# cluster_space = naive_clustering(reactions)
cluster_space = filter_by_invariants(reactions_big, ['wl1'])
cluster_space = cluster_filtered_reactions(cluster_space)
end_time = time.process_time()

print(f"Time taken: {end_time - start_time:.4f} seconds")
print(f"Total reaction centers: {len(reactions_big)}")
print(f"Number of clusters: {len(cluster_space)}")

 

# Plot a specific partition (e.g., the fourth partition)
# plot_partition(partition, partition_index=3)
# plot_representatives(cluster_space)



# Define the list of invariants to test
invariants_list = [
    ['vertex_count'],
    ['edge_count'],
    ['degree_sequence'],
    ['lex_node_sequence'],
    ['algebraic_connectivity'],
    ['rank'],
    ['wl1'],
    ['wl2'],
    ['wl3'],
    ['lex_node_sequence', 'edge_count']
]

# Measure the clustering time for each set of invariants
# results = measure_clustering_time(reactions, invariants_list)
# results_with_neighbors = measure_clustering_time(reactions_with_neighbors, invariants_list)
results_big = measure_clustering_time(reactions_big, invariants_list)
results_big_with_neighbors = measure_clustering_time(reactions_big_with_neighbors, invariants_list)

# Create a DataFrame from the results
# df = pd.DataFrame(results)
# df_with_neighbors = pd.DataFrame(results_with_neighbors)
df_big = pd.DataFrame(results_big)
df_big_with_neighbors = pd.DataFrame(results_big_with_neighbors)

# df.to_csv('df', sep=',', index=False, encoding='utf-8')
# df_with_neighbors.to_csv('df_with_neighbors', sep=',', index=False, encoding='utf-8')
df_big.to_csv('df_big', sep=',', index=False, encoding='utf-8')
df_big_with_neighbors.to_csv('df_big_with_neighbors', sep=',', index=False, encoding='utf-8')


# Display the DataFrame
print(df_big)
