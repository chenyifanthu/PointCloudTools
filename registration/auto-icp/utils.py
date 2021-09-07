import copy
import numpy as np
import open3d as o3d
from math import sin, cos, atan, acos

def remove_ground(pcd, threshold=0.04):
    pcd_nogd = copy.deepcopy(pcd)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    gpoints_idx = np.abs(points[:, 2]) < threshold
    points = points[~gpoints_idx, :]
    pcd_nogd.points = o3d.utility.Vector3dVector(points)
    if colors.shape[0]:
        colors = colors[~gpoints_idx, :]
        pcd_nogd.colors = o3d.utility.Vector3dVector(colors)
    return pcd_nogd

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
    


    
if __name__ == '__main__':
    pass