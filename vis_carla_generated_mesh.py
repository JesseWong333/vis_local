import json
import pyvista as pv
from plyfile import PlyData, PlyElement
import numpy as np


# 提供的instance 有很多是0


# plydata = PlyData.read("/Users/junjiewang/Downloads/001656.ply")
plydata = PlyData.read("/Volumes/external/Downloads/000068_semantic_mesh.ply")
vertex_data = plydata.elements[0].data
face_data = plydata.elements[1].data # 每一行是一个tuple

xyz_points = np.column_stack((vertex_data['x'], vertex_data['y'], vertex_data['z']))
instance_ids = vertex_data['ObjIdx']
semantic_ids = vertex_data['ObjTag']

faces = np.vstack(face_data['vertex_indices']) # n_face, 3

mesh = pv.PolyData()
mesh.points = xyz_points
mesh.faces = np.column_stack(( np.full(len(faces), 3), faces))


# 创建颜色数据

# sem_color_lut: sem_tag -> color
carla_color_json = json.load(open("./config/carla.json"))
sem_color_lut = np.zeros((100, 3), dtype=np.float32)
for key, value in carla_color_json.items():
    sem_color_lut[int(key)] = np.array(value["Converted color"], np.float32) / 255.0

# vertex_index -> sem_tag: 就是semantic_ids

# face_index -> vertex_index: 就是faces中的第一个点就好

face_label_color = faces[:, 0]
face_label_color = sem_color_lut[semantic_ids[face_label_color]]


# 直接使用pyvista读取ply文件似乎会丢掉其中 instance ID 和 semantic ID 的信息


mesh.cell_data["color"] = face_label_color 


# 使用plotter 和 Mesh.plot() 是等同的
# plotter = pv.Plotter()
# plotter.add_mesh(mesh, scalars="color", rgba=True, preference='cell')
# plotter.show()

mesh.plot(scalars="color", rgba=True, preference='cell') # preference为field， points, 或者cell
