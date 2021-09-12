import time
import open3d as o3d
import numpy as np
from ground import *
from global_reg import *

def preprocess(pcd, maxheight):
    plane_coef = estimate_plane_ransac(np.asarray(pcd.points))
    ground_mat = get_transform_matrix_from_plane_function(plane_coef)
    pcd.transform(ground_mat)
    pcd_nogd = remove_ground(pcd)
    if maxheight > 0:
        ind = np.where(np.asarray(pcd_nogd.points)[:, 2] < maxheight)[0]
        pcd_nogd = pcd_nogd.select_by_index(ind)
    return pcd_nogd, ground_mat

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


if __name__ == '__main__':
    leica = o3d.io.read_point_cloud('data/volleyball/leica.pts')
    livox = o3d.io.read_point_cloud('data/volleyball/livox_4.pcd')
    
    t1 = time.time()
    
    maxheight = 5
    leica_nogd, leica_gmat = preprocess(leica, maxheight=maxheight)
    livox_nogd, livox_gmat = preprocess(livox, maxheight=maxheight)
    
    voxel_size = 0.5
    # leica_nogd_down = leica_nogd.voxel_down_sample(voxel_size)
    # livox_nogd_down = livox_nogd.voxel_down_sample(voxel_size)
    leica_nogd_down = leica.voxel_down_sample(voxel_size)
    livox_nogd_down = livox.voxel_down_sample(voxel_size)
    
    estimate_normals(leica_nogd_down, voxel_size*2)
    estimate_normals(livox_nogd_down, voxel_size*2)
    leica_nogd_fpfh = extract_fpfh(leica_nogd_down, voxel_size*5)
    livox_nogd_fpfh = extract_fpfh(livox_nogd_down, voxel_size*5)
    
    ransac_res = execute_global_registration(leica_nogd_down, livox_nogd_down,
                                             leica_nogd_fpfh, livox_nogd_fpfh,
                                             distance_threshold=2)
    print(ransac_res)
    # draw_registration_result(leica_nogd_down, livox_nogd_down, ransac_res.transformation)
    
    voxel_size = 0.05
    leica_nogd_down = leica_nogd.voxel_down_sample(voxel_size)
    livox_nogd_down = livox_nogd.voxel_down_sample(voxel_size)
    estimate_normals(leica_nogd_down, voxel_size*2)
    estimate_normals(livox_nogd_down, voxel_size*2)
    print('Begin refine registration...')
    refine_result = refine_registration(leica_nogd_down, livox_nogd_down, 
                                        ransac_res.transformation,
                                        distance_threshold=0.02)
    print(refine_result)
    
    t2 = time.time()
    print('Elapse: %.2f sec' % (t2-t1))
    
    draw_registration_result(leica, livox, refine_result.transformation)
    
    livox2leica_mat = np.linalg.inv(leica_gmat).dot(np.linalg.inv(refine_result.transformation)).dot(livox_gmat)
    print(livox2leica_mat)