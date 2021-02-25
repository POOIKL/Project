import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

g = nx.Graph()
plt.title("Step 0")
plt.ylim(0, 60)
plt.xlim(0,75)
nx.draw(g, nx.get_node_attributes(g, 'pos'), with_labels=True, node_size=400, alpha=1, font_size=15)
plt.axis('on')
plt.show()