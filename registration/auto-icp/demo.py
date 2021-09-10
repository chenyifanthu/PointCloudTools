import time
import random
import open3d as o3d
import numpy as np
from ransac import estimate_plane_ransac, get_transform_matrix_from_plane_function
from utils import visualize_ground, remove_ground
from fpfh import fpfh, find_correspondence, solve_pnp_ransac, Rt2T


def preprocess(pcd, maxheight):
    plane_coef = estimate_plane_ransac(np.asarray(pcd.points))
    gorund_mat = get_transform_matrix_from_plane_function(plane_coef)
    pcd.transform(gorund_mat)
    pcd_nogd = remove_ground(pcd, threshold=0.1)
    if maxheight > 0:
        ind = np.where(np.asarray(pcd_nogd.points)[:, 2] < maxheight)[0]
        pcd_nogd = pcd_nogd.select_by_index(ind)
    return pcd_nogd
        


if __name__ == '__main__':
    
    # leica = o3d.io.read_point_cloud('data/liujiao1/leica6SE.pts')
    # livox = o3d.io.read_point_cloud('data/liujiao1/6SESE.pcd')
    
    leica = o3d.io.read_point_cloud('data/volleyball/leica.pts')
    livox = o3d.io.read_point_cloud('data/volleyball/livox_4.pcd')
    
    leica_nogd = preprocess(leica, 4)
    livox_nogd = preprocess(livox, 4)
    
    leica_nogd_down = leica_nogd.voxel_down_sample(0.2)
    livox_nogd_down = livox_nogd.voxel_down_sample(0.2)
    print(leica_nogd_down)
    print(livox_nogd_down)
    # o3d.visualization.draw_geometries([leica_nogd_down, livox_nogd_down])
    # exit()
    
    leica_fpfh = fpfh(leica_nogd_down)
    livox_fpfh = fpfh(livox_nogd_down)
    
    leica_points = np.asarray(leica_nogd_down.points)
    livox_points = np.asarray(livox_nogd_down.points)
    corr_pairs1, corr_pairs2 = find_correspondence(leica_points, livox_points,
                                                   leica_fpfh, livox_fpfh,
                                                   h_threshold=0.1, 
                                                   feat_dist_threshold=10.0)
    
    R, t = solve_pnp_ransac(leica_points[:, :2], livox_points[:, :2], 
                            corr_pairs1, corr_pairs2, 
                            step=400000, distance_threshold=0.2)
    print(R, '\n', t)
    trans = Rt2T(R, t)
    leica_nogd.transform(trans)
    
    o3d.visualization.draw_geometries([leica_nogd, livox_nogd])
    
    # t2 = time.time()
    # leica_down = leica.voxel_down_sample(0.1)
    # livox_down = livox.voxel_down_sample(0.1)
    # leica_down.estimate_normals(
    #     o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
    # livox_down.estimate_normals(
    #     o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))

    # print('begin ICP')
    # result = o3d.pipelines.registration.registration_icp(
    #     leica_down, livox_down, 0.2, np.eye(4),
    #     o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    #     o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10)
    # )
    
    # t3 = time.time()
    # print(t2-t1, t3-t2, t3-t1)
    # leica.transform(result.transformation)
    
    # o3d.visualization.draw_geometries([leica_nogd, livox_nogd])