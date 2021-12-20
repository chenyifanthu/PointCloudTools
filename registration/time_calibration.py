import os
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


class LivoxStream(object):
    def __init__(self, filelist, addn=40, startidx=0):
        assert startidx >= addn - 1
        filelist = sort_file_by_basename(filelist)
        self.filelist = filelist
        self.addn = addn
        self.idx = startidx
        self.npoints = []
        self.xyz = self.initial_read()
        
    def initial_read(self):
        pcd = o3d.geometry.PointCloud()
        for i in range(self.idx-self.addn+1, self.idx+1):
            pcd_ = o3d.io.read_point_cloud(self.filelist[i])
            self.npoints.append(get_npoint(pcd_))
            pcd += pcd_
        return np.asarray(pcd.points)
    
    def next(self):
        if self.idx >= len(self.filelist):
            return False
        self.idx += 1
        pcd_ = o3d.io.read_point_cloud(self.filelist[self.idx])
        pcd_xyz = np.asarray(pcd_.points)
        self.npoints.append(get_npoint(pcd_))
        n0 = self.npoints.pop(0)
        self.xyz = np.concatenate((self.xyz[n0:,:], pcd_xyz), axis=0)
        return True

    def get_pcd(self, voxel_size=-1, max_distance=-1, color=None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.xyz)
        if voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size)
        if max_distance > 0:
            pcd = remove_far_points(pcd, max_distance)
        if color is not None:
            pcd.paint_uniform_color(color)
        return pcd

    def get_xyz(self, voxel_size=-1, max_distance=-1):
        return self.get_pcd(voxel_size, max_distance).points
    

def sort_file_by_basename(filelist):
    return sorted(filelist, key=lambda x: int(os.path.basename(x).split('.')[0]))   


def get_npoint(pcd):
    return np.asarray(pcd.points).shape[0]


def remove_far_points(pcd, threshold=50):
    points = np.asarray(pcd.points)
    rho = np.linalg.norm(points, axis=1)
    points = points[np.where(rho<threshold)]
    pcd_ = o3d.geometry.PointCloud()
    pcd_.points = o3d.utility.Vector3dVector(points)
    return pcd_


def distance_between_pc(pcd, query_tree, distance_threshold=0.4):
    xyz = np.asarray(pcd.points)
    distance, _ = query_tree.query(xyz)
    distance = distance[np.where(distance < distance_threshold)]
    return np.mean(distance)


def setup_kdtree(pcd):
    return cKDTree(np.asarray(pcd.points))
