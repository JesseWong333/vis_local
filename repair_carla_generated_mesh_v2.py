
import json
import pyvista as pv
from plyfile import PlyData, PlyElement
import numpy as np
import trimesh
import open3d as o3d
import pyvista
import mesh2sdf
from tqdm import tqdm


ranges = [-51.2, -25.6, -2, 51.2, 25.6, 4.4]

# using mesh2sdf to get watertight mesh
def mesh2sdf_DOGN(mesh, mesh_scale=0.8, size=64, level=None):
        
    if level is None:
        level = 2 / size
    
    vertices = mesh.vertices
    #
    bbmin = vertices.min(axis=0)
    bbmax = vertices.max(axis=0)
    center = (bbmax + bbmin) / 2
    scale = 2.0 * mesh_scale / (bbmax - bbmin).max()

    vertices = (vertices - center) * scale
    
    sdf, mesh = mesh2sdf.compute(
        vertices, mesh.faces, size, fix=True, level=level, return_mesh=True)
    
    mesh.vertices = mesh.vertices / scale + center
    
    # # 可视化; 
    # vista_mesh = pv.PolyData()
    # vista_mesh.points = np.array(mesh.vertices)
    # vista_mesh.faces = np.column_stack(( np.full(mesh.faces.shape[0], 3), np.array(mesh.faces)))
    # vista_mesh.plot()
    return mesh

def water_tight_and_sampling(instance_mesh, semantic_id):
    mesh = trimesh.Trimesh(vertices = instance_mesh[0],
                       faces = instance_mesh[1],)
    if mesh.is_watertight:
        return mesh
    
    # 不同的语义ID是否要用不同的参数？
    if semantic_id in [0, 1, 2, 23, 25]: # Unlabeled, Roads, SideWalks, water, Ground
        return instance_mesh
    # elif semantic_id in [3]: # Buildings
    #     # return mesh2sdf_DOGN(instance_mesh[0], instance_mesh[1])
    #     pass
    # elif semantic_id in [4, 5, 28]: # walls, fences, GuardRail(护栏)
    #     return instance_mesh
    elif semantic_id in [14, 18]: # car
        # vista_mesh = pv.PolyData()
        # vista_mesh.points = np.array(instance_mesh[0])
        # vista_mesh.faces = np.column_stack(( np.full(instance_mesh[1].shape[0], 3), np.array(instance_mesh[1])))
        # vista_mesh.plot()
    
        return mesh2sdf_DOGN(mesh)
    # elif semantic_id in [9]: # Terrain
    #     # meshfix = mf.MeshFix(instance_mesh)  # 修复效果完全不行

    #     pass
    # elif semantic_id in [24]: # RoadLine, 过滤掉
    #     return instance_mesh 
    # Static, 路边的椅子等
    
    return mesh2sdf_DOGN(mesh)
    


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

water_tight_vertices = []
water_tight_faces = []
for instance_id in tqdm(unique_instance_ids):
    mask = (instance_ids == instance_id)
    vertex_indices = np.where(mask)[0]
    index_mapping = {old: new for new, old in enumerate(vertex_indices)}
    
    masked_xyz_points = xyz_points[mask]
    
    masked_semantic_ids = semantic_ids[mask]

    face_mask = np.isin(faces, vertex_indices).all(axis=1)
    masked_faces = faces[face_mask]  

    # Create mesh for this instance
    if len(masked_faces) > 0:  
        # 对于每一个instance
        remapped_faces = np.vectorize(index_mapping.get)(masked_faces)       
        proc_ins_mesh = water_tight_and_sampling((masked_xyz_points, remapped_faces), masked_semantic_ids[0])

        # water_tight_vertices.append( np.array(proc_ins_mesh.vertices) )
        # water_tight_faces.append( np.array(proc_ins_mesh.faces) )
        
# water_tight_vertices = np.vstack(water_tight_vertices)
# water_tight_faces = np.vstack(water_tight_faces)

# vista_mesh = pv.PolyData()
# vista_mesh.points = water_tight_vertices
# vista_mesh.faces = np.column_stack(( np.full(water_tight_faces.shape[0], 3), np.array(water_tight_faces)))
# vista_mesh.plot()




