import matplotlib.pyplot as plt
import networkx as nx
import json
import numpy as np

#d_fff6a11568773d2262a3a67a4e725b1e.json
#f = open("./new_dic/"+dic_id[i],)

f = open("./new_dic/d_958a8b570f3574492cbb84bb1b1a2d2f.json",)
dct = json.load(f)

_data = np.empty(shape=[0, 2])

for data in dct.keys():
	for connection in dct[data]:
		_data = np.append(_data, [[data, connection]], axis = 0)

print(_data)

G = nx.Graph()
G.add_edges_from(_data)

pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_size = 500)
nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edges(G, pos, arrows=False)
plt.show()
