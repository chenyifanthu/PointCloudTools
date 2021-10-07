import copy
import random
import numpy as np
from numpy.core.defchararray import asarray
import open3d as o3d

# def estimate_plane_ransac(pcd, step=200, n_ransac=3, threshold=0.04, choose_rate=0.001):
#     """Find a plane function for a series of points in 3D space with the most inliers

#     Args:
#         data (2-D ndarray of floats with shape (N, 3)): coordinates of points
#         step (int, optional): Sampling times. Defaults to 100.
#         n_ransac (int, optional): The number of points used to fit the plane each time. Defaults to 3.
#         threshold (float, optional): The distance threshold used when judging the inliers. Defaults to 0.04.
#         choose_rate (float, optional): Randomly choose some points at first if the input point cloud is too large. Defaults to 0.001.

#     Returns:
#         array[4,]: parameters of plane function, [A, B, C, D] means the function is: Ax + By + Cz + D = 0 
#     """    
    
#     assert 0.0 < choose_rate <= 1.0, 'Choose Rate should be in range (0.0, 1.0]'
#     if choose_rate != 1.0:
#         ndata = data.shape[0]
#         choose_idx = random.sample(range(ndata), int(choose_rate * ndata))
#         data_ = data[choose_idx, :]
    
#     best_inlier_count = 0
#     best_coef = None
#     ndata = data_.shape[0]
#     print('Using %d points to estimate plane function' % ndata)
#     data_ = np.hstack((data_, np.ones((ndata, 1))))
#     for i in range(step):
#         choose_idx = random.sample(range(ndata), n_ransac)
#         xyz_sample = data_[choose_idx, :]
#         coef = np.linalg.svd(xyz_sample)[-1][-1, :]
#         distance = np.abs(data_.dot(coef)) / np.linalg.norm(coef[:-1])
#         inlier_idx = distance <= threshold
#         count = sum(inlier_idx)
#         if count > best_inlier_count:
#             best_inlier_count = count
#             best_coef = coef        
    
#     print('Ground points Rate: %.3f' % (best_inlier_count / ndata))
#     if best_coef[2] <= 0:
#         best_coef = -best_coef
#     return best_coef


def normalize(vec):
    return vec / np.linalg.norm(vec)


def get_transform_matrix_from_plane_function(coef):
    assert len(coef) == 4
    a, b, c, d = coef
    z_axis = normalize(np.array([a, b, c]))
    x_axis = normalize(np.array([b, -a, 0]))
    y_axis = np.cross(z_axis, x_axis)
    R = np.vstack((x_axis, y_axis, z_axis))
    t = np.array([[0, 0, d*z_axis[0]/a]])
    T = np.block([[R, t.T], [np.zeros((1, 3)), 1]])
    return T


def remove_ground(pcd, threshold=0.04):
    pcd_down = pcd.voxel_down_sample(0.05)
    plane_model, inliers = pcd_down.segment_plane(distance_threshold=threshold, 
                                                  ransac_n=3, num_iterations=200)
    [a, b, c, d] = plane_model
    print("    - plane model: %.2fx+%.2fy+%.2fz+%.2f=0" % (a, b, c, d))
    print("    - ground point rate: %.4f" % (len(inliers) / np.asarray(pcd_down.points).shape[0]))
    trans_mat = get_transform_matrix_from_plane_function(plane_model)
    pcd_down.transform(trans_mat)
    pcd_nogd = pcd_down.select_by_index(inliers, invert=True)
    return pcd_nogd, trans_mat


if __name__ == '__main__':
    import open3d as o3d
    import time
    file_path = r"C:\Users\win10\Desktop\data\20210923\DCE.ply"
    print('Reading file: %s' % file_path)
    pcd = o3d.io.read_point_cloud(file_path)
    print(help(o3d.geometry.PointCloud))
    # remove_ground(pcd)
    # points = np.asarray(pcd.points)
    # start = time.time()
    # coef = estimate_plane_ransac(points, choose_rate=0.001)
    # end = time.time()
    # print('Elapse: %.2f sec' % (end - start))