import open3d as o3d
import numpy as np
from ransac import estimate_plane_ransac
from utils import get_transform_matrix_from_plane_function

if __name__ == '__main__':
    
    pcd = o3d.io.read_point_cloud('data/20210808/leica.ply')
    pcd_down = pcd.voxel_down_sample(0.05)
    
    plane_coef = estimate_plane_ransac(np.asarray(pcd.points))
    trans_mat = get_transform_matrix_from_plane_function(plane_coef)
    
    print(trans_mat)
    pcd_down.transform(trans_mat)
    o3d.io.write_point_cloud('data/20210808/leica_gd_down.ply', pcd_down)
    o3d.visualization.draw_geometries([pcd_down])
    
    # a, b, c, d = 1, 5, 2, 4
    # x, y = np.mgrid[-2:2:0.01, -2:2:0.01]
    # x, y = x.flatten(), y.flatten()
    # z = (-d-a*x-b*y)/ c
    # points = np.vstack((x, y, z)).T
    # points_ = np.hstack((points, np.ones((len(z), 1))))
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    
    # coef = estimate_plane_ransac(points)
    # trans_mat = get_transform_matrix_from_plane_function(coef)
    # print(np.asarray(pcd.points))
    # pcd.transform(trans_mat)
    # print(np.asarray(pcd.points))
    # # print(trans_mat.dot(points_.T).T)
    