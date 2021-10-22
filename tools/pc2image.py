import cv2
import copy
import open3d as o3d
import numpy as np
from math import tan, pi
from scipy.interpolate import griddata


def select_points_by_fov(pcd, hfov, vfov, scale=1):
    points = np.asarray(pcd.points)
    idx = np.where((points[:, 0] > 0) 
                   & (np.arctan(abs(points[:, 1])/points[:, 0]) < scale * hfov/2)
                   & (np.arctan(abs(points[:, 2])/points[:, 0]) < scale * vfov/2))
    return pcd.select_by_index(idx[0])


def clip(img):
    img *= 255
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    return img


def interpolate(xy, rgb, xi, method='nearest'):
    assert xy.shape[0] == rgb.shape[0]
    grid_rgb = [griddata(xy, rgb[:, c], xi, method=method) for c in range(2, -1, -1)]
    grid_rgb = [clip(channel) for channel in grid_rgb]
    grid_rgb_ = cv2.merge(grid_rgb)
    return grid_rgb_


def add_mask(points_uv, image, kernel_size):
    fimh, fimw = image.shape[0], image.shape[1]
    mask = np.zeros((fimh, fimw, 3), dtype=np.float64)
    points_uv = points_uv.astype(int)
    idx = np.where((points_uv[:, 0] >= 0) & (points_uv[:, 0] < fimw)
                   & (points_uv[:, 1] >= 0) & (points_uv[:, 1] < fimh))    
    mask[points_uv[idx[0], 1], points_uv[idx[0], 0]] = 1
    g = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, g)
    return image * mask


def pointcloud_to_image(pcd_, fx, fy, hfov, vfov, rotx, roty, rotz, 
                        interpolation="nearest", kernel_size=3):
    
    """This function helps to generate an image from a point cloud with a specific perspective.
    The principle of this algorithm is to project the color points onto an imaging plane perpendicular to the x axis.
    And can mask out the pixels that are not projected on the image.

    Args:
        pcd_ (open3d.geometry.PointCloud): A pointcloud with RGB colors, such as the data export from Leica.
        fx (int): The focal length of the x-axis when generating the image.
        fy (int): The focal length of the y-axis when generating the image.
        hfov (float): The horizontal angle of view of the image.
        vfov (float): The horizontal angle of view of the image.
        rotx (float): The initial rotation angle of the point cloud around the x axis.
        roty (float): The initial rotation angle of the point cloud around the y axis.
        rotz (float): The initial rotation angle of the point cloud around the z axis.
        interpolation (str, optional): Method of interpolation, one of \{'linear', 'nearest', 'cubic'\}. Defaults to "nearest".
        kernel_size (int, optional): The kernal size of morphological closing operation when adding mask, if you DONT want to add mask, set it to 0. Defaults to 3.

    Returns:
        image (array): 3-D ndarray of uint8 with shape (H, W, 3). The generated image.
        intrinsics (array): 2-D ndarray of float with shape (3, 3). The Camera intrinsics of the image.
    """

    assert pcd_.has_colors
    
    pcd = copy.deepcopy(pcd_)
    R = o3d.geometry.get_rotation_matrix_from_xyz([rotx, roty, rotz])
    pcd.rotate(R)
    pcd_select = select_points_by_fov(pcd, hfov, vfov)

    points = np.asarray(pcd_select.points)
    colors = np.asarray(pcd_select.colors)

    fimw = int(2 * fx * tan(hfov/2))
    fimh = int(2 * fy * tan(vfov/2))
    grid_u, grid_v= np.meshgrid(range(fimw), range(fimh))
    grid_u, grid_v = grid_u.flatten(), grid_v.flatten()

    u = - fx * points[:, 1] / points[:, 0] + fimw / 2
    v = - fy * points[:, 2] / points[:, 0] + fimh / 2
    points_uv = np.array([u, v]).T
    grid_rgb = interpolate(points_uv, colors, np.array([grid_u, grid_v]).T, method=interpolation)
    image = grid_rgb.reshape((fimh, fimw, 3))
    
    if kernel_size > 0:
        image = add_mask(points_uv, image, kernel_size)
    
    intrinsics = np.array([[fx, 0, fimw/2], [0, fy, fimh/2], [0, 0, 1]])
    
    return image, intrinsics
    
    