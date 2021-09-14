import time
import open3d as o3d
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree
from ground import *
from global_reg import *

def preprocess(pcd, maxheight):
    plane_coef = estimate_plane_ransac(np.asarray(pcd.points))
    ground_mat = get_transform_matrix_from_plane_function(plane_coef)
    pcd.transform(ground_mat)
    pcd_nogd = remove_ground(pcd, 0.1)
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


def registration_ransac_based_on_feature_matching(source, target, source_fpfh, target_fpfh, 
                                                  n_neighbors=1, h_threshold=0.1, distance_threshold=0.2):
    source_fpfh = source_fpfh.data.T
    target_fpfh = target_fpfh.data.T
    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)
    
    t = time.time()    
    search_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
    search_model.fit(source_fpfh)
    ind = search_model.kneighbors(target_fpfh, return_distance=False)
    print(time.time() - t)

    select_ind = ind[:, 0]

    z_src = source_points[select_ind, 2]
    z_tar = target_points[:, 2]

    hvalid_ind = np.abs(z_src-z_tar) < h_threshold
    corr1 = ind[hvalid_ind, 0]
    corr2 = np.array(range(target_points.shape[0]))[hvalid_ind]
    print(corr1, corr2)
    
    
    ransac_res = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source, target, 
        o3d.utility.Vector2iVector(np.vstack((corr1, corr2)).T), distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3, [
           o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
           o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
               distance_threshold)
       ], o3d.pipelines.registration.RANSACConvergenceCriteria(10000000, 500))
    
    return ransac_res
    
    
if __name__ == '__main__':
    leica = o3d.io.read_point_cloud('data/volleyball/leica.pts')
    livox = o3d.io.read_point_cloud('data/volleyball/livox_4.pcd')
    
    t1 = time.time()
    
    maxheight = 4
    leica_nogd, leica_gmat = preprocess(leica, maxheight=maxheight)
    livox_nogd, livox_gmat = preprocess(livox, maxheight=maxheight)
    
    voxel_size = 0.5
    leica_nogd_down = leica_nogd.voxel_down_sample(voxel_size)
    livox_nogd_down = livox_nogd.voxel_down_sample(voxel_size)
    print(leica_nogd_down)
    print(livox_nogd_down)
    # leica_nogd_down = leica.voxel_down_sample(voxel_size)
    # livox_nogd_down = livox.voxel_down_sample(voxel_size)
    
    estimate_normals(leica_nogd_down, voxel_size*2)
    estimate_normals(livox_nogd_down, voxel_size*2)
    leica_nogd_fpfh = extract_fpfh(leica_nogd_down, voxel_size*5)
    livox_nogd_fpfh = extract_fpfh(livox_nogd_down, voxel_size*5)
    
    ransac_res = registration_ransac_based_on_feature_matching(leica_nogd_down, livox_nogd_down,
                                                               leica_nogd_fpfh, livox_nogd_fpfh,
                                                               h_threshold=voxel_size*1, 
                                                               distance_threshold=voxel_size*1.5)
    print(ransac_res)
    draw_registration_result(leica_nogd_down, livox_nogd_down, ransac_res.transformation)