import open3d as o3d
import numpy as np
import time
import random
from tqdm import tqdm


def Rt2T(R, t):
    T = np.zeros((4, 4))
    T[3, 3] = 1
    if R.shape[0] == 2:
        T[:2, :2] = R
        T[:2, 3:] = t
        T[2, 2] = 1
    elif R.shape[0] == 3:
        T[:3, :3] = R
        T[:3, 3:] = t
    else:
        raise ValueError()
    
    return T

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
            
       
def find_correspondence(xyz1, xyz2, feat1, feat2, h_threshold=0.05, feat_dist_threshold=20.0):
    sort_idx = np.argsort(xyz2[:, 2])
    xyz2 = xyz2[sort_idx, :]
    feat2 = feat2[sort_idx, :]
    corr_pairs1, corr_pairs2 = [], []
    for i in tqdm(range(xyz1.shape[0])):
        left, right = valid_height_in_range(xyz2[:, 2], 
                                            xyz1[i, 2] - h_threshold, 
                                            xyz1[i, 2] + h_threshold)
        
        if left != -1:
            distances = np.linalg.norm(feat1[i, :] - feat2[left:right, :], axis=1)
            if np.min(distances) < feat_dist_threshold:
                corr_pairs1.append(i)
                corr_pairs2.append(sort_idx[left+np.argmin(distances)])
            
    return np.array(corr_pairs1), np.array(corr_pairs2)



def fpfh(pcd):
    
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=30))

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=100))
    
    return fpfh.data.T

def solve_pnp_ransac(source, target, 
                     corr_pairs1, corr_pairs2, 
                     ransac_n=3, step=5000, 
                     distance_threshold=0.5):
    assert len(corr_pairs1) == len(corr_pairs2)
    print('Correspondence size: ', len(corr_pairs1))
    dim = source.shape[1]
    Rs = np.zeros((step, dim, dim))
    ts = np.zeros((step, dim, 1))
    valid_idx = []
    for i in tqdm(range(step)):
        choose_id = random.sample(range(len(corr_pairs1)), ransac_n)
        source_ = source[corr_pairs1[choose_id], :].T
        target_ = target[corr_pairs2[choose_id], :].T
        center1 = np.mean(source_, axis=1, keepdims=True)
        center2 = np.mean(target_, axis=1, keepdims=True)
        W = (source_ - center1).dot((target_ - center2).T)
        u, _, vt = np.linalg.svd(W)
        R = u.dot(vt)
        t = center2 - R.dot(center1)
        Rs[i, :, :] = R
        ts[i, :, :] = t
        if R[0, 0] == R[1, 1]:
            valid_idx.append(i)
    valid_idx = np.array(valid_idx)
    target_hat = Rs[valid_idx, :, :].dot(source[corr_pairs1, :].T) + ts[valid_idx, :, :]
    # target_hat = Rs.dot(source[corr_pairs1, :].T) + ts
    distances = np.linalg.norm(target[corr_pairs2, :].T - target_hat, axis=1)
    inliers = np.sum(distances < distance_threshold, axis=1)
    best_idx = np.argmax(inliers)
    print('Best inlier number: %d' % inliers[best_idx])
    # return Rs[best_idx, :, :], ts[best_idx, :, :]
    return Rs[valid_idx[best_idx], :, :], ts[valid_idx[best_idx], :, :]
    
def solve_pnp_ransac_2(source, target, corr_pairs1, corr_pairs2, distance_threshold):
    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source, target,
        o3d.utility.Vector2iVector(np.vstack((corr_pairs1, corr_pairs2)).T), 
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9), 
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    
    return result.transformation
    

if __name__ == '__main__':
    pcd = o3d.io.read_point_cloud('data/20210808/leica_down.ply')
    # o3d.visualization.draw_geometries([pcd])
    xyz = np.asarray(pcd.points)
    feat = fpfh(pcd)
    corr_pairs1, corr_pairs2 = find_correspondence(xyz, xyz.copy(), feat, feat.copy())
    solve_pnp_ransac(xyz[:, :2], xyz[:, :2], corr_pairs1, corr_pairs2)
    # print(corr_pairs2)

