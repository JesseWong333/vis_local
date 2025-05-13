
# convolutional occupancy

# 检查 ply文件的点云是否填充， 无，只包含表面
import numpy as np
from plyfile import PlyData, PlyElement
import pyvista as pv

plydata = PlyData.read("/Users/junjiewang/Downloads/pointcloud0.ply")
vertex_data = plydata.elements[0].data
face_data = plydata.elements[1].data # 每一行是一个tuple

xyz_points = np.column_stack((vertex_data['x'], vertex_data['y'], vertex_data['z']))

# 切开查看内部是否填充了
x_mean = np.mean(xyz_points[:, 1])
mask = xyz_points[:, 1] > x_mean
xyz_points = xyz_points[mask]

point_cloud = pv.PolyData(xyz_points)
plotter = pv.Plotter()
plotter.add_mesh(point_cloud, 
                 scalars=xyz_points[:, 2],
                 cmap='viridis', 
                 point_size=5)
plotter.show()


