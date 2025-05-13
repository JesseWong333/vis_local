import numpy as np
import pyvista as pv

points = np.load("/Users/junjiewang/Downloads/points_2000_no_color.npy")

point_cloud = pv.PolyData(points)
# 添加基于Z坐标的标量值用于颜色映射
point_cloud['height'] = points[:, 2]  
plotter = pv.Plotter()
plotter.add_mesh(
    point_cloud, 
    scalars='height', 
    cmap='coolwarm',  # 使用coolwarm颜色映射
    point_size=15,     # 适当调整点大小
    render_points_as_spheres=True
)
plotter.show()

