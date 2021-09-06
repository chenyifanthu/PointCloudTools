import copy
import numpy as np
import open3d as o3d
from math import sin, cos, atan, acos
from global_reg import *

# def get_rotation_matrix(x_rot, y_rot, z_rot):
#     rx = np.array([[1, 0, 0], 
#                    [0, cos(x_rot), -sin(x_rot)], 
#                    [0, sin(x_rot), cos(x_rot)]])
#     ry = np.array([[cos(y_rot), 0, sin(y_rot)], 
#                    [0, 1, 0],
#                    [-sin(y_rot), 0, cos(y_rot)]])
#     rz = np.array([[cos(z_rot), -sin(z_rot), 0], 
#                    [sin(z_rot), cos(z_rot), 0],
#                    [0, 0, 1]])
#     return rx.dot(ry).dot(rz)

def remove_ground(pcd, threshold=0.04):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    gpoints_idx = np.abs(points[:, 2]) < threshold
    points = points[~gpoints_idx, :]
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors.shape[0]:
        colors = colors[~gpoints_idx, :]
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def visualize_ground(pcd, threshold=0.04, fade=0.4):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    gpoints_idx = np.abs(points[:, 2]) < threshold
    colors[gpoints_idx, :] = 1 - fade + fade * colors[gpoints_idx, :]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
    
def register_leica_with_livox(leica_pcd, livox_pcd,
                              voxel_size_large=0.5,
                              voxel_size_small=0.05,
                              max_distance=10):

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
    
    print('粗配准中......')
    ransac_res = execute_global_registration(leica_down_large, 
                                             livox_down_large, 
                                             leica_down_large_fpfh, 
                                             livox_down_large_fpfh, 
                                             max_distance)
    
    print(ransac_res)
    # draw_registration_result(leica_down_large, livox_down_large, ransac_res.transformation)
    
    print('精配准中......')
    refine_res = refine_registration(leica_down_small, 
                                     livox_down_small, 
                                     ransac_res.transformation, 
                                     0.1 * max_distance)
    
    print(refine_res)
    draw_registration_result(leica_down_small, livox_down_small, refine_res.transformation)
    
    return refine_res.transformation


    
if __name__ == '__main__':
    pass