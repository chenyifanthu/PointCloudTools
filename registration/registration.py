import copy
import time
import open3d as o3d
from .ground import *
from preprocess.read_data import read_point_cloud


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
    
def preprocess(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    return pcd_down
    
def extract_feature(pcd, voxel_size):
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100))
    return pcd_fpfh


def global_registration(source_down, target_down, 
                        source_fpfh, target_fpfh, 
                        distance_threshold):
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    return result


def adaptive_icp(source, target, init_trans, threshold_list):
    if type(threshold_list) is not list:
        threshold_list = [threshold_list]
    best_fitness = float('inf')
    best_result = None
    for distance_threshold in threshold_list:
        result = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, init_trans,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        fitness = result.inlier_rmse / distance_threshold
        print("    - Threshold: %f | fitness: %.4f" % (distance_threshold, fitness))
        if fitness < best_fitness:
            best_fitness = fitness
            best_result = result
    return best_result, best_fitness


def register_two_pointclouds(source, target, down_sizes=[0.5, 0.2, 0.1]):
    
    ## Estimate ground transform matrix
    print("=== Estimate ground function and remove ground ===")
    start = time.time()
    ground_threshold = 0.1
    source_nogd, source_gd_mat = remove_ground(source, threshold=ground_threshold)
    target_nogd, target_gd_mat = remove_ground(target, threshold=ground_threshold)
    source.transform(source_gd_mat)
    target.transform(target_gd_mat)
    print("    - Elapse: %.3f sec" % (time.time() - start))
    
    
    ## Preprocess data
    print("=== Downsample and Calculate FPFH feature ===")
    start = time.time()
    source_down_list, target_down_list = [], []
    source_nogd_down_list, target_nogd_down_list = [], []
    for voxel_size in down_sizes:
        print("    - voxel_size: %f" % voxel_size)
        source_down_list.append(preprocess(source, voxel_size))
        target_down_list.append(preprocess(target, voxel_size))
        source_nogd_down_list.append(preprocess(source_nogd, voxel_size))
        target_nogd_down_list.append(preprocess(target_nogd, voxel_size))
        
    source_fpfh = extract_feature(source_down_list[0], down_sizes[0])
    target_fpfh = extract_feature(target_down_list[0], down_sizes[0])
    source_nogd_fpfh = extract_feature(source_nogd_down_list[0], down_sizes[0])
    target_nogd_fpfh = extract_feature(target_nogd_down_list[0], down_sizes[0])
    print("    - Elapse: %.3f sec" % (time.time() - start))
    
    
    ## Rough Registration
    print("=== Rough Registration ===")
    start = time.time()
    withground = False
    rough_threshold = 5
    if withground:
        rough_result = global_registration(source_down_list[0], target_down_list[0],
                                           source_fpfh, target_fpfh, rough_threshold)
    else:
        rough_result = global_registration(source_nogd_down_list[0], target_nogd_down_list[0],
                                           source_nogd_fpfh, target_nogd_fpfh, rough_threshold)
    print("    - ", rough_result)
    print("    - Elapse: %.3f sec" % (time.time() - start))
    draw_registration_result(source_down_list[-1], target_down_list[-1], rough_result.transformation)
    
    
    ## Mid Registration
    print("=== Middle Registration ===")
    start = time.time()
    step = 5
    mid_threshold = [i / step * rough_threshold for i in range(2, step+1)]
    mid_result1, mid_fitness1 = adaptive_icp(source_down_list[1], target_down_list[1], 
                                             rough_result.transformation, mid_threshold)
    mid_result2, mid_fitness2 = adaptive_icp(source_nogd_down_list[1], target_nogd_down_list[1], 
                                             rough_result.transformation, mid_threshold)
    mid_result = mid_result1 if mid_fitness1 < mid_fitness2 else mid_result2
    print("    - ", mid_result)
    print("    - Elapse: %.3f sec" % (time.time() - start))
    draw_registration_result(source_down_list[-1], target_down_list[-1], mid_result.transformation)
    
    
    ## Refine Registration
    print("=== Refine Registration ===")
    start = time.time()
    refine_threshold = down_sizes[-1] * 0.5
    refine_result, _ = adaptive_icp(source_down_list[2], target_down_list[2], 
                                    mid_result.transformation, refine_threshold)
    print("    - ", refine_result)
    print("    - Elapse: %.3f sec" % (time.time() - start))
    draw_registration_result(source_down_list[-1], target_down_list[-1], refine_result.transformation)
    
    return np.linalg.inv(target_gd_mat) @ refine_result.transformation @ source_gd_mat

if __name__ == '__main__':
    source = read_point_cloud(r"C:\Users\win10\Desktop\data\20210923\DCNW.ply")
    target = read_point_cloud(r"C:\Users\win10\Desktop\data\20210923\DCE.ply")
    result = register_two_pointclouds(source, target)
    print(result)
    