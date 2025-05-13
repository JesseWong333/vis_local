import pyvista as pv
from glob import glob
import numpy as np

# iou 的内部是有点的, 墙壁有一点点厚度

# 每个文件有 100000 100k 个点， 10个文件共存储了 100w 个点

file_list = glob("/Volumes/external/Downloads/00000000/points_iou/*.npz")

valid_points_l = []

for file_path in file_list:
    points_dict = np.load(file_path)
    points = points_dict['points']
    occupancies = points_dict['occupancies']
    occupancies = np.unpackbits(occupancies)[:points.shape[0]]

    mask = occupancies != 0
    valid_points_l.append(points[mask])

valid_points = np.concatenate(valid_points_l, axis=0).astype(np.float32)

# center = valid_points.mean(axis=0)
# valid_points = valid_points[(valid_points - center)[:,1] > 0]

point_cloud = pv.PolyData(valid_points)
point_cloud['color'] = valid_points[:, 2]
plotter = pv.Plotter()
plotter.add_mesh(point_cloud, scalars='color', cmap='viridis')
plotter.show()
