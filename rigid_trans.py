import copy
import numpy as np
import open3d as o3d

mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)  # red: x, green: y, blue: z
mesh0 = copy.deepcopy(mesh)
mesh1 = copy.deepcopy(mesh)
mesh2 = copy.deepcopy(mesh)
mesh3 = copy.deepcopy(mesh)
mesh4 = copy.deepcopy(mesh)

T = (0, -0.68, 0)
mesh0.translate(T)

T = (0.478, -0.68, 0.198)
R = mesh.get_rotation_matrix_from_xyz((0, -np.pi / 4, 0))
mesh1.translate(T)
mesh1.rotate(R)

T = (0.676, -0.68, 0.676)
R = mesh.get_rotation_matrix_from_xyz((0, -np.pi / 2, 0))
mesh2.translate(T)
mesh2.rotate(R)

T = (-0.676, -0.68, 0.676)
R = mesh.get_rotation_matrix_from_xyz((0, np.pi / 2, 0))
mesh3.translate(T)
mesh3.rotate(R)

T = (-0.478, -0.68, 0.198)
R = mesh.get_rotation_matrix_from_xyz((0, np.pi / 4, 0))
mesh4.translate(T)
mesh4.rotate(R)

o3d.visualization.draw_geometries([mesh, mesh0, mesh1, mesh2, mesh3, mesh4])
