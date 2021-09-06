import open3d as o3d
from sklearn.neighbors import NearestNeighbors, DistanceMetric, BallTree
import numpy as np
import time
from tqdm import tqdm
    

def valid_height_in_range(height, minheight, maxheight):
    left, right = 0, len(height) - 1
    while left <= right:
        mid = (left + right) // 2
        if height[mid] < minheight:
            left = mid + 1
        else:
            right = mid - 1

    if left >= len(height) or height[left] > maxheight:
        return -1, -1
    
    left_idx = left
    right = len(height) - 1
    while left <= right and left < len(height):
        mid = (left + right) // 2
        if height[mid] < maxheight:
            left = mid + 1
        else:
            right = mid - 1
    right_idx = left

    return left_idx, right_idx
            
       
def find_correspondence(xyz1, xyz2, feat1, feat2, h_threshold=0.05):
    sort_idx = np.argsort(xyz2[:, 2])
    xyz2 = xyz2[sort_idx, :]
    feat2 = feat2[sort_idx, :]
    corr_list = []
    for i in tqdm(range(xyz1.shape[0])):
        left, right = valid_height_in_range(xyz2[:, 2], 
                                            xyz1[i, 2] - h_threshold, 
                                            xyz1[i, 2] + h_threshold)
        
        if left != -1:
            distances = np.linalg.norm(feat1[i, :] - feat2[left:right, :], axis=1)
            corr_list.append((i, sort_idx[left+np.argmin(distances)]))
            
    return corr_list


def fpfh(pcd):
    
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=30))

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=100))
    
    return fpfh.data.T

if __name__ == '__main__':

    pcd = o3d.io.read_point_cloud('data/20210808/leica_down.ply')
    # o3d.visualization.draw_geometries([pcd])
    xyz = np.asarray(pcd.points)
    feat = fpfh(pcd)
    corr_list = find_correspondence(xyz, xyz.copy(), feat, feat.copy())
    print(corr_list)

