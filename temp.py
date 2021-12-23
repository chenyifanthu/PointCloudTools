import math
import numpy as np
import open3d as o3d


pcd = o3d.io.read_point_cloud('./data/leica.ply')
coord = o3d.geometry.TriangleMesh.create_coordinate_frame()

theta = math.pi/4
R = np.array([[math.cos(theta), math.sin(theta), 0],
                [0, 0, -1],
                [-math.sin(theta), math.cos(theta), 0]])

coord_rot = coord.rotate(R)

o3d.visualization.draw_geometries([pcd, coord_rot])

