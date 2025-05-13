import numpy as np
import pyvista as pv

points = np.load("/Volumes/external/Downloads/points_2000_no_color.npy")

point_cloud = pv.PolyData(points)
point_cloud['color'] = points[:, 2]
plotter = pv.Plotter()
plotter.add_mesh(point_cloud, scalars='color', cmap='viridis')
plotter.show()
