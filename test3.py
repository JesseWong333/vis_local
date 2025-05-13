
# normx = np.array(direction) / np.linalg.norm(direction) 或者说这个叫法向量

import pyvista as pv
mesh = pv.ParametricEllipsoid(xradius=1, yradius=2, zradius=3, direction=[0., 0., -1.])
mesh.plot(color='w', smooth_shading=True)

