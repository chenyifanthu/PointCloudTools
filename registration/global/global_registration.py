import open3d as o3d
from utils import *
import copy
import time

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
    
def register_leica_with_livox(leica_pcd, livox_pcd,
                              voxel_size_large=0.5,
                              voxel_size_small=0.2,
                              max_distance=5):

    print('降采样中...')
    leica_down_large = leica_pcd.voxel_down_sample(voxel_size_large)
    leica_down_small = leica_pcd.voxel_down_sample(voxel_size_small)
    
    livox_down_large = livox_pcd.voxel_down_sample(voxel_size_large)
    livox_down_small = livox_pcd.voxel_down_sample(voxel_size_small)
    
    print('估计法线并计算特征中...')
    leica_down_large = estimate_normals(leica_down_large, voxel_size_large * 2)
    leica_down_large_fpfh = extract_fpfh(leica_down_large, voxel_size_large * 5)
    
    livox_down_large = estimate_normals(livox_down_large, voxel_size_large * 2)
    livox_down_large_fpfh = extract_fpfh(livox_down_large, voxel_size_large * 5)
    
    leica_down_small = estimate_normals(leica_down_small, voxel_size_small * 2)
    livox_down_small = estimate_normals(livox_down_small, voxel_size_small * 2)
    
    # print('快速配准中...')
    # fast_result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
    #     leica_down_large, livox_down_large, leica_down_large_fpfh, livox_down_large_fpfh,
    #     o3d.pipelines.registration.FastGlobalRegistrationOption(
    #         maximum_correspondence_distance=voxel_size_large * 0.5,
    #         decrease_mu=True, iteration_number=64))
    
    print('粗配准中......')
    ransac_res = execute_global_registration(leica_down_large, 
                                             livox_down_large, 
                                             leica_down_large_fpfh, 
                                             livox_down_large_fpfh, 
                                             max_distance)
    
    print(ransac_res)
    draw_registration_result(leica_pcd, livox_pcd, ransac_res.transformation)
    
    # print('精配准中......')
    # refine_res = refine_registration(leica_down_small, 
    #                                  livox_down_small, 
    #                                  ransac_res.transformation, 
    #                                  0.2 * max_distance)
    
    # print(refine_res)
    
    
    # draw_registration_result(leica_pcd, livox_pcd, ransac_res.transformation)
    
    # return refine_res.transformation

leica_pcd = o3d.io.read_point_cloud('data/volleyball/leica.pts')
livox_pcd = o3d.io.read_point_cloud('data/volleyball/livox_4.pcd')
start = time.time()
register_leica_with_livox(leica_pcd, livox_pcd)
print('Elapse: %.2f sec' % (time.time()-start))