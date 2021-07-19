import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial import Voronoi, voronoi_plot_2d

rng = np.random.default_rng()
points = rng.random((100,2))

print(points)

test_num = 1

#points = np.load('matrices/reward_matrix_'+str(test_num)+'.npy', allow_pickle=False)

vor = Voronoi(points)

fig = voronoi_plot_2d(vor)

fig = voronoi_plot_2d(vor, show_vertices=True, line_colors='orange',
                line_width=2, line_alpha=0.6, point_size=2)

plt.show()


