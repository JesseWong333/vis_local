
# import numpy as np
# import pyvista as pv

# # 定义3D高斯的参数
# mu = np.array([1, 2, 3])  # 中心
# cov = np.array([[4, 1, 0], [1, 3, 0], [0, 0, 1]])  # 协方差矩阵

# # 计算特征值和特征向量
# eigenvals, eigenvecs = np.linalg.eig(cov)

# # 3σ椭球的半轴长度
# radii = 3 * np.sqrt(eigenvals)  # [a, b, c]

# # 主轴方向（取最大特征值对应的特征向量）
# main_axis = eigenvecs[:, np.argmax(eigenvals)]
# direction = main_axis / np.linalg.norm(main_axis)  # 归一化

# # 创建椭球
# ellipsoid = pv.ParametricEllipsoid(
#     xradius=radii[0],
#     yradius=radii[1],
#     zradius=radii[2],
#     direction=direction,
#     center=mu
# )

# # 可视化
# plotter = pv.Plotter()
# plotter.add_mesh(ellipsoid, opacity=0.8)
# plotter.add_axes()
# plotter.show()

# -------------------------------------------------------
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation

# 椭球参数
xradius = 1.5  # x轴半长
yradius = 5.0  # y轴半长
zradius = 0.1  # z轴半长
quaternion = [0.5, 0.5, 0, 0]  # 四元数 (x, y, z, w)
center = [1, 2, 3]  # 椭球中心（可选）

# 1. 创建椭球（默认在原点，无旋转）
ellipsoid = pv.ParametricEllipsoid(
    xradius=xradius,
    yradius=yradius,
    zradius=zradius,
)

# 2. 四元数 -> 旋转矩阵
rotation = Rotation.from_quat(quaternion)
rotation_matrix = rotation.as_matrix()

# 3. 构造变换矩阵（旋转 + 平移）
transform_matrix = np.eye(4)
transform_matrix[:3, :3] = rotation_matrix  # 设置旋转部分
transform_matrix[:3, 3] = center  # 设置平移部分

# 4. 应用变换
ellipsoid.transform(transform_matrix, inplace=True)

# 5. 可视化
plotter = pv.Plotter()
plotter.add_mesh(ellipsoid, opacity=0.8)
plotter.add_axes()
plotter.show()

# -------------------------------------------------------
# import numpy as np

import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation

# 椭球参数
xradius, yradius, zradius = 1.5, 5.0, 0.1  # 半轴长度
euler_angles = [0.0, 0, 1.2]  # 欧拉角 (rx, ry, rz)，单位弧度
center = [1, 2, 3]  # 椭球中心

# 1. 创建默认椭球（无旋转，中心在原点）
ellipsoid = pv.ParametricEllipsoid(
    xradius=xradius,
    yradius=yradius,
    zradius=zradius
)

# 2. 欧拉角 -> 旋转矩阵
rotation = Rotation.from_euler('xyz', euler_angles)  # 注意旋转顺序！
rotation_matrix = rotation.as_matrix()

# 3. 构造 4x4 变换矩阵（旋转 + 平移）
transform_matrix = np.eye(4)
transform_matrix[:3, :3] = rotation_matrix  # 旋转部分
transform_matrix[:3, 3] = center           # 平移部分

# 4. 应用变换
ellipsoid.transform(transform_matrix, inplace=True)

# 5. 可视化
plotter = pv.Plotter()
plotter.add_mesh(ellipsoid, color="red", opacity=0.8)
plotter.add_axes()
plotter.show()