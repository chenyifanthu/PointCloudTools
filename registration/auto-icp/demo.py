import time
import open3d as o3d
import numpy as np
from global_reg import estimate_normals, extract_fpfh
from ransac import estimate_plane_ransac, get_transform_matrix_from_plane_function
from utils import visualize_ground, remove_ground
from fpfh import fpfh, find_correspondence, solve_pnp_ransac, solve_pnp_ransac_2, Rt2T

def register_leica_with_livox(leica_pcd, livox_pcd,
                              voxel_size_large=0.5,
                              voxel_size_small=0.05):
    
    plane_coef = estimate_plane_ransac(np.asarray(leica.points))
    leica_gorund_mat = get_transform_matrix_from_plane_function(plane_coef)
    leica.transform(leica_gorund_mat)
    leica_nogd = remove_ground(leica)
    
    plane_coef = estimate_plane_ransac(np.asarray(livox.points))
    livox_gorund_mat = get_transform_matrix_from_plane_function(plane_coef)
    livox.transform(livox_gorund_mat)
    livox_nogd = remove_ground(livox)
    
    leica_down_large = leica_pcd.voxel_down_sample(voxel_size_large)
    leica_down_small = leica_pcd.voxel_down_sample(voxel_size_small)
    livox_down_large = livox_pcd.voxel_down_sample(voxel_size_large)
    livox_down_small = livox_pcd.voxel_down_sample(voxel_size_small)

    leica_down_large = estimate_normals(leica_down_large, voxel_size_large * 2)
    livox_down_large = estimate_normals(livox_down_large, voxel_size_large * 2)
    leica_down_small = estimate_normals(leica_down_small, voxel_size_small * 2)
    livox_down_small = estimate_normals(livox_down_small, voxel_size_small * 2)
    
    leica_down_large_fpfh = extract_fpfh(leica_down_large, voxel_size_large * 5)
    livox_down_large_fpfh = extract_fpfh(livox_down_large, voxel_size_large * 5)
    
    plane_coef = estimate_plane_ransac(np.asarray(leica_down_small.points), choose_rate=0.1)
    leica_gorund_mat = get_transform_matrix_from_plane_function(plane_coef)
    leica_down_large.transform(leica_gorund_mat)
    leica_nogd = remove_ground(leica)
    


if __name__ == '__main__':
    
    leica = o3d.io.read_point_cloud('data/20210808/leica.ply')
    livox = o3d.io.read_point_cloud('data/20210808/SouthWest.pcd')
    
    t1 = time.time()
    plane_coef = estimate_plane_ransac(np.asarray(leica.points))
    leica_gorund_mat = get_transform_matrix_from_plane_function(plane_coef)
    leica.transform(leica_gorund_mat)
    leica_nogd = remove_ground(leica)
    
    plane_coef = estimate_plane_ransac(np.asarray(livox.points))
    livox_gorund_mat = get_transform_matrix_from_plane_function(plane_coef)
    livox.transform(livox_gorund_mat)
    livox_nogd = remove_ground(livox)
    
    leica_nogd_down = leica_nogd.voxel_down_sample(0.4)
    livox_nogd_down = livox_nogd.voxel_down_sample(0.4)
    print(leica_nogd_down, '\n', livox_nogd_down)
    
    leica_fpfh = fpfh(leica_nogd_down)
    livox_fpfh = fpfh(livox_nogd_down)
    
    leica_points = np.asarray(leica_nogd_down.points)
    livox_points = np.asarray(livox_nogd_down.points)
    corr_pairs1, corr_pairs2 = find_correspondence(leica_points, livox_points,
                                                   leica_fpfh, livox_fpfh,
                                                   h_threshold=0.05, 
                                                   feat_dist_threshold=10.0)
    
    R, t = solve_pnp_ransac(leica_points[:, :2], livox_points[:, :2], 
                            corr_pairs1, corr_pairs2, 
                            step=200000, distance_threshold=0.2)
    print(R, '\n', t)
    trans = Rt2T(R, t)
    leica.transform(trans)
    # o3d.visualization.draw_geometries([leica, livox])
    
    t2 = time.time()
    leica_down = leica.voxel_down_sample(0.05)
    livox_down = livox.voxel_down_sample(0.05)
    leica_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    livox_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    print('begin ICP')
    result = o3d.pipelines.registration.registration_icp(
        leica_down, livox_down, 0.2, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10)
    )
    
    t3 = time.time()
    print(t2-t1, t3-t2, t3-t1)
    leica.transform(result.transformation)
    o3d.visualization.draw_geometries([leica, livox])