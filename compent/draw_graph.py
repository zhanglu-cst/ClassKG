import networkx as nx
from matplotlib import pyplot as plt


def draw(g):
    nx.draw(g.to_networkx(), with_labels = True)
    plt.show()
