# 测试open3d voxelize时内部的点是否有值,  结论 体素化，采样都是表面
# 测试Trimesh： sample.volume_mesh可以在内部采样，但非water tight的mesh，会采样不到; 使用rejection sampling
# 测试pyvista：体素化似乎可以vox.points即可以得到，即使缺了面也可以; 要取vox.points； 但非water tight的mesh，体素化不准确
import pyvista as pv
import numpy as np
import open3d as o3d
import trimesh

# Create a cube mesh
mesh_o3d = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)


mesh = pv.PolyData()
# vertices = np.asarray([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
# triangles = np.asarray([[0, 1, 2], [0, 2, 3]])

vertices = np.array(mesh_o3d.vertices)
triangles = np.array(mesh_o3d.triangles)[3:,:]

mesh.points = vertices
mesh.faces = np.column_stack(( np.full(len(triangles), 3), triangles))

mesh.plot()


vox = pv.voxelize(mesh, density=0.13, check_surface=False)  # UnstructuredGrid 类型

vox_centers = np.array(vox.points)  # 中心点不会超出范围

point_cloud = pv.PolyData(vox_centers)
plotter = pv.Plotter()
plotter.add_mesh(point_cloud)
plotter.show()
