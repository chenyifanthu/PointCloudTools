import open3d as o3d
import numpy as np
from ransac import estimate_plane_ransac, get_transform_matrix_from_plane_function
from utils import visualize_ground, register_leica_with_livox, remove_ground

if __name__ == '__main__':
    
    leica = o3d.io.read_point_cloud('data/20210808/leica.ply')
    plane_coef = estimate_plane_ransac(np.asarray(leica.points))
    leica_gorund_mat = get_transform_matrix_from_plane_function(plane_coef)
    leica.transform(leica_gorund_mat)
    leica_nogd = remove_ground(leica)
    
    livox = o3d.io.read_point_cloud('data/20210808/SouthEast.pcd')
    plane_coef = estimate_plane_ransac(np.asarray(livox.points))
    livox_gorund_mat = get_transform_matrix_from_plane_function(plane_coef)
    livox.transform(livox_gorund_mat)
    livox_nogd = remove_ground(livox)
    
    trans = register_leica_with_livox(leica_nogd, livox_nogd)
    print(trans)
    
    
    # pcd_down = pcd.voxel_down_sample(0.05)
    
    # plane_coef = estimate_plane_ransac(np.asarray(pcd.points))
    # trans_mat = get_transform_matrix_from_plane_function(plane_coef)
    
    # print(trans_mat)
    # pcd_down.transform(trans_mat)
    
    # pcd_down = o3d.io.read_point_cloud('data/20210808/leica_down.ply')
    # visualize_ground(pcd_down, threshold=0.3)
    