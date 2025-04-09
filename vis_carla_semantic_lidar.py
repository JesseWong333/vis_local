import os
import numpy as np
import glob
import json
import open3d as o3d
from plyfile import PlyData, PlyElement
import pyvista as pv

import json

# 有趣的点：语义分割：摩托车和车上的人是两个类别
# 实例分割中，似乎只有运动的人，车，路灯等被创建为类别，其余的都是一个instance ID

def creat_color_maps(sem_color_dict):
    # copied from laserscan.py
    
     # make semantic colors
    max_sem_key = 0
    for key, data in sem_color_dict.items():
      key = int(key)
      if key + 1 > max_sem_key:
        max_sem_key = key + 1
    sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
    for key, value in sem_color_dict.items():
      key = int(key)
      sem_color_lut[key] = np.array(value["Converted color"], np.float32) / 255.0

    # make instance colors
    max_inst_id = 100000
    inst_color_lut = np.random.uniform(low=0.0,
                                            high=1.0,
                                            size=(max_inst_id, 3))
    # force zero to a gray-ish color
    inst_color_lut[0] = np.full((3), 0.1)
    return sem_color_lut, inst_color_lut
def colorize(sem_color_lut, inst_color_lut, sem_label, inst_label):
    """ Colorize pointcloud with the color of each semantic label
    """
    sem_label_color = sem_color_lut[sem_label]
    sem_label_color = sem_label_color.reshape((-1, 3))

    inst_label_color = inst_color_lut[inst_label]
    inst_label_color = inst_label_color.reshape((-1, 3))
    return sem_label_color, inst_label_color
    
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

with open('./config/carla.json', 'r', encoding='utf-8') as f:
    semantic_color_map = json.load(f)

sem_color_lut, inst_color_lut = creat_color_maps(semantic_color_map)
    
file_path = "/Users/junjiewang/Downloads/out/semantic"
ego_pose = "/Users/junjiewang/Downloads/out/ego_transformation.json"

# lidar_range = [0, -25.6, -2, 51.2, 25.6, 4.4] # 这里的0是车辆雷达传感器的位置的0
lidar_range = [-150, -150, -15, 150, 150, 15]

ego_to_world_trans = np.array(json.loads(open(ego_pose).read())['961']) # ego车辆的位姿
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
pcd = mask_ego_points(pcd)

point_cloud = pv.PolyData(pcd[:, :3])
point_cloud['colors'] = colorize(sem_color_lut, inst_color_lut, pcd[:, 5].astype(np.uint32), pcd[:, 4].astype(np.uint32))[0]  #   0, semantic_label, 1 instance_label
plotter = pv.Plotter()
plotter.add_mesh(point_cloud, scalars='colors', rgb=True, point_size=1)
plotter.show()

