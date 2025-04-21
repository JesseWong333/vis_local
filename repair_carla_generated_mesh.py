
import json
import pyvista as pv
from plyfile import PlyData, PlyElement
import numpy as np
import trimesh
import open3d as o3d
import pyvista
import mesh2sdf

# carla 产生的 Mesh 不一定是闭合的，我需要将将这些点; 闭合的Mesh叫watertight mesh
# 
# ObjIdx是carla自身给的，只包括车，交通信号灯，行人等
# ObjTag是carla给的，包含了所有的semantic标签
# ActorIndex是按照component给的标签， 有的可能会被拆分


# 修复的方式有非常多种，
# 1，体素化
# 2. 泊松重建
# 3. 孔洞填充

# 1) 对于路面等信息： 向下扩充至范围内
# 2）对于植被树木等信息：使用某种方式进行闭合；pymeshfix效果不行


# 非闭合的物体也可以体素化，所以一定要体素化吗？


ranges = [-51.2, -25.6, -2, 51.2, 25.6, 4.4]

# 方式1: 泊松重建修复， # 效果不好或者难以封闭
def PoissonReconFix(points, faces):
    # 使用Open3D进行泊松重建
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    pcd = mesh.sample_points_poisson_disk(number_of_points=10000)
    o3d.visualization.draw_geometries([pcd])
    # 2. 估计法向量（泊松重建必需）
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.2, max_nn=30))

    # 3. 泊松重建
    mesh_poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=10, linear_fit=True)
         
    mesh_poisson.compute_vertex_normals()  # 为更好的可视化效果计算法线
    mesh_poisson.paint_uniform_color([0.7, 0.7, 0.7])
    o3d.visualization.draw_geometries([mesh_poisson])
    
    
    return mesh_poisson.vertices, mesh_poisson.triangles

# 方式1: pymeshfix. 修复效果完全不行
# meshfix = mf.MeshFix(instance_mesh) 
# meshfix.repair(verbose=True)  

def voxelize_mesh(points, faces, voxel_size=5): # 
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size)  # 单位是厘米
    
    voxels = voxel_grid.get_voxels()
    voxel_centers = np.asarray([voxel_grid.get_voxel_center_coordinate(v.grid_index) for v in voxels])
    # voxel_centers: n_voxel, 3
    # 把这个切开，来观察内部
    # x_mean = np.mean(voxel_centers[:, 0])
    # mask = voxel_centers[:, 0] > x_mean
    # voxel_centers = voxel_centers[mask]
    
    pdata = pv.PolyData(voxel_centers)

    cube = pv.Cube(x_length=4.9, y_length=4.9, z_length=4.9)
    pc = pdata.glyph(scale=False, geom=cube, orient=False)
    pc.plot(show_axes=True)
    pass

def voxelize_mesh_pyvista(points, faces, voxel_size=5):
    mesh = pv.PolyData()
    mesh.points = points
    mesh.faces = np.column_stack(( np.full(len(faces), 3), faces))
    vox = pv.voxelize(mesh, density=voxel_size, check_surface=False)  # UnstructuredGrid 类型
    voxel_centers = np.array(vox.points)  # 中心点不会超出范围
    
    # 把这个切开，来观察内部
    # x_mean = np.mean(voxel_centers[:, 0])
    # mask = voxel_centers[:, 0] > x_mean
    # voxel_centers = voxel_centers[mask]
    
    # 非water_tight_mesh，使用voxelize_mesh_pyvista好像也不准
    pdata = pv.PolyData(voxel_centers)

    cube = pv.Cube(x_length=4.9, y_length=4.9, z_length=4.9)
    pc = pdata.glyph(scale=False, geom=cube, orient=False)
    pc.plot(show_axes=True)
# 这个用了处理绿植明显很好
def mesh2sdf_DOGN(vertices, faces):
    
    mesh_scale = 0.8
    size = 64 # 64格？
    level = 2 / size
    
    # 必须要normalize吗
    bbmin = vertices.min(axis=0)
    bbmax = vertices.max(axis=0)
    center = (bbmax + bbmin) / 2
    scale = 2.0 * mesh_scale / (bbmax - bbmin).max()

    vertices = (vertices - center) * scale
    
    sdf, mesh = mesh2sdf.compute(
        vertices, faces, size, fix=True, level=level, return_mesh=True)
    
    mesh.vertices = mesh.vertices / scale + center
    
    # 可视化; 
    vista_mesh = pv.PolyData()
    vista_mesh.points = np.array(mesh.vertices)
    vista_mesh.faces = np.column_stack(( np.full(mesh.faces.shape[0], 3), np.array(mesh.faces)))
    vista_mesh.plot()
    pass

# 一个可能的做法：water_tight_mesh 里面填充； 非water_tight_mesh只进行表面体素化  
def water_tight_mesh(instance_mesh, semantic_id):
    mesh = trimesh.Trimesh(vertices = instance_mesh[0],
                       faces = instance_mesh[1],)
    if mesh.is_watertight:
        return instance_mesh
    
    # 仅处理非watertight的
    return mesh2sdf_DOGN(instance_mesh[0], instance_mesh[1])

    # 将不同的类别分类
    # if semantic_id in [0, 1, 2, 23, 25]: # Unlabeled, Roads, SideWalks, water, Ground
    #     return instance_mesh
    # elif semantic_id in [3]: # Buildings
    #     return instance_mesh
    # elif semantic_id in [4, 5, 28]: # walls, fences, GuardRail(护栏)
    #     return instance_mesh
    # elif semantic_id in [14, 18]: # car
    #     return mesh2sdf_DOGN(instance_mesh[0], instance_mesh[1])
    # elif semantic_id in [9]: # Terrain
    #     # meshfix = mf.MeshFix(instance_mesh)  # 修复效果完全不行
    #     # meshfix.repair(verbose=True)
    #     # return meshfix.mesh
    #     return mesh2sdf_DOGN(instance_mesh[0], instance_mesh[1])
    # elif semantic_id in [24]: # RoadLine, 过滤掉
    #     return instance_mesh 
    # Static, 路边的椅子等
    return instance_mesh


plydata = PlyData.read("/Users/junjiewang/Downloads/003916_mesh.ply")
vertex_data = plydata.elements[0].data
face_data = plydata.elements[1].data # 每一行是一个tuple

xyz_points = np.column_stack((vertex_data['x'], vertex_data['y'], vertex_data['z']))
instance_ids = vertex_data['ObjIdx']
semantic_ids = vertex_data['ObjTag']
actor_ids = vertex_data['ActorIndex']  # 使用ActorIndex作为instance ID

# ############################################################
# 如果actor_ids 对应的instance_ids是一样的，就将Actor_ids合并成一样的
meraged_isds = actor_ids.copy()
unique_instance_ids = np.unique(instance_ids)

for instance_id in unique_instance_ids:
    if instance_id == 0:
        continue
    mask = (instance_ids == instance_id)
    meraged_isds[mask] = meraged_isds[mask][0]

# ############################################################

instance_ids = meraged_isds

faces = np.vstack(face_data['vertex_indices']) # n_face, 3
# 对每一个instance创建watertight mesh

unique_instance_ids = np.unique(instance_ids)

for instance_id in unique_instance_ids:
    mask = (instance_ids == instance_id)
    vertex_indices = np.where(mask)[0]
    index_mapping = {old: new for new, old in enumerate(vertex_indices)}
    
    masked_xyz_points = xyz_points[mask]
    
    masked_semantic_ids = semantic_ids[mask]

    face_mask = np.isin(faces, vertex_indices).all(axis=1)
    masked_faces = faces[face_mask]  

    # Create mesh for this instance
    if len(masked_faces) > 0:
        # ############################################################
        # 对于每一个instance
        remapped_faces = np.vectorize(index_mapping.get)(masked_faces)
        # instance_mesh = pv.PolyData()
        # instance_mesh.points = masked_xyz_points
        # instance_mesh.faces = np.column_stack((np.full(len(masked_faces), 3), remapped_faces))
        # instance_mesh.plot() 
        
        # # ############################################################
        proc_ins_mesh = water_tight_mesh((masked_xyz_points, remapped_faces), masked_semantic_ids[0])
        # proc_ins_mesh.plot()
        
        pass




