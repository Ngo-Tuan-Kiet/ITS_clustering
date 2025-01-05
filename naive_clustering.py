import pickle
import networkx as nx
import matplotlib.pyplot as plt
from synutility.SynIO.data_type import load_from_pickle
from synutility.SynVis.graph_visualizer import GraphVisualizer
from synutility.SynAAM.misc import get_rc
import networkx.algorithms.isomorphism as iso
import time


def is_isomorphic_to_partition(rc, partition):
    """Checks if a reaction center (rc) is isomorphic to any set in the partition and returns index of set."""
    for i, subset in enumerate(partition):
        if nx.is_isomorphic(
            subset[0], rc,
            node_match=iso.categorical_node_match(["charge", "element"], [1, 'H']),
            edge_match=iso.categorical_edge_match("order", (1.0, 1.0))
        ):
            return i
    return None


# • Implement a simple clustering algorithm:
# – Input: Set of reactions R
# – Out: Partition Q ∶= Q1,Q2,...,QN as defined above
# – for each reaction r in R:
#     let rc be the reaction center of r
#     for each Qi in Q:
#         if rc is isomorphic to a representative of Qi, add Qi = Qi ∪ rc; break;
#     if rc could not be added to any set, add a new set Qj = {rc} to Q
def naive_cluster_reaction_centers(reaction_centers):
    """Clusters reaction centers into partitions based on isomorphism."""
    partition = []
    for rc in reaction_centers:
        index = is_isomorphic_to_partition(rc, partition)
        if index is not None: # Add to existing partition
            partition[index].append(rc)
        else:
            partition.append([rc])
    return partition

# We now want to apply pre-filters to roughly group reactions before applying WP2 on sub-
# groups.
# • From the lecture we know that graph invariants do not change between isormorphic
# graphs, granting them their name.
# • We therefore can use them to group our reactions:
# – If the invariant is identical, reactions centers may be isomorphic, so they are added
# to the same cluster.
# – If the invariant is different, reactions centers cannot be isomorphic, so they must
# appear in different clusters.
#     • Modify WP2 such that graphs are clustered not by isomorphism, but by the invariant.
#     • Apply isomorphism clustering to further subdivide each invariant cluster to the final
#     isomorphism cluster set Q.
# • Test various graph invariants of your liking, including at least vertex and edge counts,
# vertex degrees, algebraic connectivity and rank (look here for inspiration https://en.wikipedia.org/wiki/Category:Graph_invariants).

def get_graph_invariants(graph):
    """Returns a dictionary of graph invariants."""
    # unoptimized (computes all invarinats if only one is needed)
    invariants = {}
    invariants['vertex_count'] = graph.number_of_nodes()
    invariants['edge_count'] = graph.number_of_edges()
    invariants['degree_sequence'] = sorted([d for n, d in graph.degree()], reverse=True)
    invariants['lex_node_sequence'] = sorted([graph.nodes[n]['element'] for n in graph.nodes], reverse=True)
    # invariants['algebraic_connectivity'] = nx.algebraic_connectivity(graph) # bugggy (value that should be the same are not because of small numerical differences)
    # invariants['rank'] = nx.linalg.graphmatrix.adjacency_matrix(graph).todense().rank()
    return invariants


def compare_graph_invariants(graph1, graph2, invariants):
    """Compares two graphs based on a list of invariants."""
    invariants1 = get_graph_invariants(graph1)
    invariants2 = get_graph_invariants(graph2) # unoptimized
    for inv in invariants:
        if invariants1[inv] != invariants2[inv]:
            return False
    return True


def cluster_by_invariant(reaction_centers, relevant_invariants):
    """Clusters reaction centers based on graph invariants given by a list of invariants you want to use."""
    partition = []
    test = 0
    for rc in reaction_centers:
        test += 1
        all_invariants = get_graph_invariants(rc)
        index = None
        for i, subset in enumerate(partition):
            if compare_graph_invariants(rc, subset[0], relevant_invariants): # unoptimized
                index = i
                break
        if index is not None:
            partition[index].append(rc)
        else:
            partition.append([rc])
            print(test)
            print(f"New partition added: {all_invariants}")

    print(f"Number of partitions after filtering: {len(partition)}")
    return partition


def naive_cluster_filtered_reaction_centers(filtered_reaction_centers):
    """Clusters reaction centers that are already partitioned by invariants further into partitions based on isomorphism."""
    partition = []
    for invariant_cluster in filtered_reaction_centers:
        sub_partition = naive_cluster_reaction_centers(invariant_cluster)
        partition.extend(sub_partition)
    return partition



def plot_partition(partition, partition_index=0):
    """Plots a specific partition."""
    fig, ax = plt.subplots(figsize=(15, 10))
    vis = GraphVisualizer()
    for rc in partition[partition_index]:
        vis.plot_its(rc, ax, use_edge_color=True)
        plt.show()
        fig, ax = plt.subplots(figsize=(15, 10))  # Create a new figure for the next plot


def plot_represntatives(partition):
    """Plots a representative of each partition."""
    fig, ax = plt.subplots(figsize=(15, 10))
    vis = GraphVisualizer()
    for subset in partition:
        vis.plot_its(subset[0], ax, use_edge_color=True)
        plt.show()
        fig, ax = plt.subplots(figsize=(15, 10))  # Create a new figure for the next plot


def main():
    filepath = 'ITS_graphs.pkl'
    data = load_from_pickle(filepath)
    reaction_centers = [get_rc(d['ITS']) for d in data]

    start_time = time.process_time()
    # partition = naive_cluster_reaction_centers(reaction_centers)
    partition = cluster_by_invariant(reaction_centers, ['vertex_count', 'edge_count', 'degree_sequence', 'lex_node_sequence'])
    partition = naive_cluster_filtered_reaction_centers(partition)
    end_time = time.process_time()


    print(f"Total reaction centers: {len(reaction_centers)}")
    print(f"Number of partitions: {len(partition)}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    # Plot a specific partition (e.g., the fourth partition)
    # plot_partition(partition, partition_index=3)

if __name__ == "__main__":
    main()