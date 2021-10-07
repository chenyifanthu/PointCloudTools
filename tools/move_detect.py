import open3d as o3d
import numpy as np

def compare_points(pcd1, pcd2, distance_threshold=0.1):
    """ Pick out the index of the points of one point cloud that can find the corresponding point in another point cloud.

    Args:
        pcd1 (open3d.geometry.PointCloud): A point cloud to be traversed.
        pcd1 (open3d.geometry.PointCloud): A point cloud to be compared.
        distance_threshold (float, optional): Distance threshold of a pair of corresponding points. Defaults to 0.1.

    Returns:
        inlier (1-d array): Index of points which CAN find correspondence.
        outlier (1-d array): Index of points which CANNOT find correspondence.
    """
    pcd_tree1 = o3d.geometry.KDTreeFlann(pcd1)
    pcd_tree2 = o3d.geometry.KDTreeFlann(pcd2)
    diff_list = np.ones(len(pcd1.points)) * -1
    for i in range(len(pcd1.points)):
        if diff_list[i] == -1:
            dist = np.sqrt(pcd_tree2.search_knn_vector_3d(pcd1.points[i], 1)[2])
            idx = pcd_tree1.search_radius_vector_3d(pcd1.points[i], abs(dist[0]-distance_threshold))[1]
            diff_list[idx[:]] = 0 if dist[0] > distance_threshold else 1
    inlier = np.where(diff_list==1)[0]
    outlier = np.where(diff_list==0)[0]
    return inlier, outlier


def combine_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)
    inlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
    outlier_cloud.paint_uniform_color([1, 0, 0])
    return inlier_cloud + outlier_cloud
    
