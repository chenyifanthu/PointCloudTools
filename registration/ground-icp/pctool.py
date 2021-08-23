# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 10:21:25 2019

@author: 机智的pzy
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:33:17 2019

@author: 机智的pzy
"""
import numpy as np
import open3d as o3d
import math 


def display_inlier_outlier(cloud, ind, change_color = False):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)  # invert利用索引反选

    print("Showing outliers (gray) and inliers (red): ")
    outlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    inlier_cloud.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud]) 
    if change_color:
        return inlier_cloud+outlier_cloud

def colmap2pc(name):
    f_read  = open(name + ".txt")
    while True:
        line = f_read.readline().split()
        if line[1] == 'Number':
            n = np.int(line[4][:-1])
            break
    pc = np.zeros((n,6))
    for i in range(n):
        line = f_read.readline().split()
        for j in range(6):
            pc[i,j] = np.double(line[j+1])
    return pc


def pc_transform(pc,trans):
    n = len(pc[:,0])
    pc_k = np.zeros((n,4))#kuozhan
    pc_k[:,0:3] = pc[:,0:3]
    pc_k[:,3] = 1
    pc_k = np.dot(trans,pc_k.T).T
    pc[:,0:3] = pc_k[:,0:3]
    return pc

def cut_pc(pc,rate = 1):
    n = len(pc[:,0])
    point = pc[:,:3]
    mean = point.mean(axis=0)
    l2 = pow(point - mean, 2);
    var = np.mean(l2, axis = 0)
    del_id = []
    for i in range(n):
        if l2[i][0] + l2[i][1] + l2[i][2] > rate*(var[0] + var[1] + var[2]) :
            del_id.append(i)
    pc = np.delete(pc, del_id ,0)
    n = np.shape(pc)[0]
    return pc

def print_mat(mat):
    mat = np.array(mat)
    print("[", end='')
    for i in range(4):
        print("[", end='')
        for j in range(4):
            print(round(mat[i][j],10), end='')
            if j != 3:
                print(",", end='')
        print("]", end='')
        if i != 3:
            print(",")
    print("]", end='')

def generate_rotation_mat(axis,theta,printmat=False):
    a = np.array([0,0,0])
    a[axis] = 1
    mat = np.matrix([[ a[0]+a[1]*math.cos(theta)+a[2]*math.cos(theta),-a[2]*math.sin(theta),-a[1]*math.sin(theta),0],
                     [ a[2]*math.sin(theta), a[0]*math.cos(theta)+a[1]+a[2]*math.cos(theta),-a[0]*math.sin(theta),0],
                     [ a[1]*math.sin(theta), a[0]*math.sin(theta), a[0]*math.cos(theta)+a[1]*math.cos(theta)+a[2],0],
                     [ 0, 0, 0, 1]])
    if printmat == True:
        print_mat(mat)
    return mat
    
def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    # pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down = o3d.geometry.voxel_down_sample(pcd,voxel_size)
    
    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    o3d.geometry.estimate_normals(
        pcd_down,o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def change_type(a,b,name,visual = False,down_sample = 0):
    target = o3d.io.read_point_cloud(name +'.'+ a)
    if down_sample != 0:
        target = o3d.geometry.voxel_down_sample(target, down_sample)
        o3d.io.write_point_cloud(name + '_d.' + b,target)
    else:
        o3d.io.write_point_cloud(name +'.'+ b,target)
    if visual == True:
        o3d.visualization.draw_geometries([target])

def regular_mat(translation = False,size = 1,printmat = False,points = None):
    if points.all() == None:
        points = np.array([[1.579147 ,1.478089, -1.052884,1],
                          [-0.878491, 1.729535 ,-1.031050,1],
                          [1.252102, -2.254328, -1.051975,1]])
    deg = math.atan((points[0,0]-points[1,0])/(points[0,1]-points[1,1]))
    mat1 = generate_rotation_mat(2,deg)
    points = np.dot(mat1,points.T).T
    deg = -math.atan((points[0,2]-points[1,2])/(points[0,1]-points[1,1]))
    mat2 = generate_rotation_mat(0,deg)
    points = np.dot(mat2,points.T).T
    deg = -math.atan((points[0,2]-points[2,2])/(points[0,0]-points[2,0]))
    mat3 = generate_rotation_mat(1,deg)
    points = np.dot(mat3,points.T).T
    mat = np.dot(mat3,np.dot(mat2,mat1))
    # mat[2,3] = -points[0,2]
    if translation == True:
        mat[0:3,3] = -points[0,0:3].T
    mat[0:3,0:3] = mat[0:3,0:3]*size
    if printmat == True:
        print_mat(mat)
    return mat

# if __name__ == '__main__':
#     mat = regular_mat()
#     name = "./dalitang/dalitang" + '1'
#     target = o3d.io.read_point_cloud(name  + '.ply')
#     target.transform(mat)
#     o3d.io.write_point_cloud(name +'n.ply',target)
#     o3d.io.write_point_cloud(name +'n.pts',target)

