import os
import numpy as np
import glob
import json
import open3d as o3d
from plyfile import PlyData, PlyElement

def save_ply(points, filename):
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('CosAngle', 'f4'), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]
    vertex = np.empty(points.shape[0], dtype=dtype)
    vertex['x'] = points[:, 0]
    vertex['y'] = points[:, 1]
    vertex['z'] = points[:, 2]
    vertex['CosAngle'] = points[:, 3]
    vertex['ObjIdx'] = points[:, 4]
    vertex['ObjTag'] = points[:, 5]
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(filename)
def project_points(point_cloud, transform_matrix):
    homogeneous_point_cloud = np.concatenate((point_cloud, np.ones((point_cloud.shape[0], 1))), axis=1) # N x 4
    transformed_point_cloud = (transform_matrix @ homogeneous_point_cloud.T).T
    return transformed_point_cloud[:, :3]

def filter_points_by_range(points, limit_range):
    
    mask =    (points[:, 0] > limit_range[0]) \
            & (points[:, 0] < limit_range[3]) \
            & (points[:, 1] > limit_range[1]) \
            & (points[:, 1] < limit_range[4]) \
            & (points[:, 2] > limit_range[2]) \
            & (points[:, 2] < limit_range[5])

    points = points[mask]
    return points

def mask_ego_points(points):

    mask = (points[:, 0] >= -1.95) & (points[:, 0] <= 2.95) \
           & (points[:, 1] >= -1.5) & (points[:, 1] <= 1.5)
    points = points[np.logical_not(mask)]
    return points

file_path = "/Users/junjiewang/Downloads/out/semantic"
ego_pose = "/Users/junjiewang/Downloads/out/ego_transformation.json"

lidar_range = [0, -25.6, -2, 51.2, 25.6, 4.4] # 这里的0是车辆雷达传感器的位置的0


ego_to_world_trans = np.array(json.loads(open(ego_pose).read())['461832']) # ego车辆的位姿
lidar_to_ego_trans = np.array([[1, 0, 0, -0.5], 
                               [0, 1, 0, 0], 
                               [0, 0, 1, 1.85],
                               [0, 0, 0, 1]]) # lidar相对于车辆的位姿

lidar_to_world_trans = lidar_to_ego_trans @ ego_to_world_trans
world_to_lidar_trans = np.linalg.inv(lidar_to_world_trans)

file_list = glob.glob(file_path + "/*.ply") 

pcd_list = []
for file in file_list:
    plydata = PlyData.read(file)
    pcd = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z'], plydata['vertex']['CosAngle'], plydata['vertex']['ObjIdx'], plydata['vertex']['ObjTag']]).T # N*6
    pcd_list.append(pcd)

pcd = np.concatenate(pcd_list, axis=0)   

projected_pcd = project_points(pcd[:, :3], world_to_lidar_trans)
pcd[:, :3] = projected_pcd

pcd = filter_points_by_range(pcd, lidar_range)
pcd = mask_ego_points(pcd) # ego位置的点云怎么重建mesh

# 对每一个物体重建GT Mesh， 这里的object有路灯，车等
unique_object_index = np.unique(pcd[:, 4]).astype(int)

for obj_idx in unique_object_index:
    if obj_idx == 0:
        # 0是背景，包含建筑路面等
        continue
    obj_mask = pcd[:, 4] == obj_idx 
    obj_pcd = pcd[obj_mask]
    
    # create mesh
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(obj_pcd[:, :3])
    # o3d.visualization.draw_geometries([o3d_pcd])

    # o3d.geometry.PointCloud.estimate_normals_search_radius()

    o3d_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.05, max_nn=30))

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        o3d_pcd, 
        depth=12,     # 重建深度（增加细节但需要更多计算）
        width=0,     # 0表示自动确定
        scale=1.1,   # 网格间距与样本间距的比例
        linear_fit=False
    )

    # vertices_to_remove = densities < np.quantile(densities, 0.01)
    # mesh.remove_vertices_by_mask(vertices_to_remove)

    # 可视化结果
    mesh.compute_vertex_normals()  # 为更好的可视化效果计算法线
    mesh.paint_uniform_color([0.7, 0.7, 0.7])
    o3d.visualization.draw_geometries([mesh])

    # save_ply(pcd, "test.ply")
