import math
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from synutility.SynIO.data_type import load_from_pickle
from synutility.SynVis.graph_visualizer import GraphVisualizer
from synutility.SynAAM.misc import get_rc

data = load_from_pickle('ITS_graphs.pkl')
rc = get_rc(data[0]['ITS'])

fig, ax = plt.subplots(2, 1, figsize=(15, 10))
vis = GraphVisualizer()
vis.plot_its(data[0]['ITS'], ax[0], use_edge_color=True)
vis.plot_its(rc, ax[1], use_edge_color=True)

plt.show()