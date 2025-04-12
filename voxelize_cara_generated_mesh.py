# 体素化里面是空的

import json
import pyvista as pv
from plyfile import PlyData, PlyElement
import numpy as np
import pymeshfix as mf
import open3d as o3d

plydata = PlyData.read("/Users/junjiewang/Downloads/000103_semantic_mesh.ply")
vertex_data = plydata.elements[0].data
face_data = plydata.elements[1].data # 每一行是一个tuple

xyz_points = np.column_stack((vertex_data['x'], vertex_data['y'], vertex_data['z']))
instance_ids = vertex_data['ObjIdx']
semantic_ids = vertex_data['ObjTag']
actor_ids = vertex_data['ActorIndex']  # 使用ActorIndex作为instance ID

faces = np.vstack(face_data['vertex_indices']) # n_face, 3

mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(xyz_points)
mesh.triangles = o3d.utility.Vector3iVector(faces)

voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, 20) 


voxels = voxel_grid.get_voxels()
voxel_centers = np.asarray([voxel_grid.get_voxel_center_coordinate(v.grid_index) for v in voxels])

ranges = [-51.2, -25.6, -2, 51.2, 25.6, 4.4]
# Define filter bounds (adjust these values based on your needs)
x_min, x_max = -5120, 5120  # Example X-axis bounds # 单位只厘米
y_min, y_max = -2560, 2540  # Example Y-axis bounds
z_min, z_max = -200, 440  # Example Z-axis bounds

# Create mask for filtering
mask = (
    (voxel_centers[:, 0] >= x_min) & (voxel_centers[:, 0] <= x_max) &
    (voxel_centers[:, 1] >= y_min) & (voxel_centers[:, 1] <= y_max) &
    (voxel_centers[:, 2] >= z_min) & (voxel_centers[:, 2] <= z_max)
)
# open3d可视化不好，但是体素化比较快，使用pyvista体素化


voxel_inds = voxel_centers[mask]


pdata = pv.PolyData(voxel_inds)

cube = pv.Cube(x_length=19.5, y_length=19.5, z_length=19.5)
pc = pdata.glyph(scale=False, geom=cube, orient=False)
pc.plot(show_axes=True)



