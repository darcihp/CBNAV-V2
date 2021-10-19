'''
import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial import Voronoi, voronoi_plot_2d

rng = np.random.default_rng()
points = rng.random((100,2))

print(points)

test_num = 1

points = np.load('matrices/reward_matrix_'+str(test_num)+'.npy', allow_pickle=False)

vor = Voronoi(points)

fig = voronoi_plot_2d(vor)

fig = voronoi_plot_2d(vor, show_vertices=True, line_colors='orange',
                line_width=2, line_alpha=0.6, point_size=2)

plt.show()
'''


import json
  
# Opening JSON file
f = open('0e91907cda3c5a42ee44d2672ba2cad8.json',)

f_d = open('data.txt', 'w')



# returns JSON object as 
# a dictionary
data = json.load(f)
  
# Iterating through the json
# list
for i in data['verts']:
    print(i)
    f_d.write(str(i[0]))
    f_d.write(" ")
    f_d.write(str(i[1]))
    f_d.write('\n')
  
# Closing file
f.close()
f_d.close()

