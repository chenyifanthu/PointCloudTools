# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 17:24:57 2021

@author: PZY
"""
# source = o3d.io.read_point_cloud("./xqlk/extract20/0.pcd")
# o3d.io.write_point_cloud("./xqlk/groud.ply",pcd)
# o3d.visualization.draw_geometries([source, target])
# pcd, ind = pcd.remove_radius_outlier(nb_points=5, radius=1)
# pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0) 
# voxel_down_pcd = source.voxel_down_sample(voxel_size=0.02)  # 下采样，大小为2cm
# pcd.points = o3d.utility.Vector3dVector(np_pc)
# np_pc = np.asarray(pcd.points)
from sklearn.decomposition import PCA
import pctool as pt
import open3d as o3d
import numpy as np
import random
import copy
import time
import math
# import sys
# sys.setrecursionlimit(1000000)
def tr_points2plant(points):
    ver1 = points[0,:] - points[1,:]
    ver2 = points[2,:] - points[1,:]
    A = (ver1[1]*ver2[2]-ver2[1]*ver1[2])/(ver2[1]*ver1[0]-ver1[1]*ver2[0])
    B = (ver1[0]*ver2[2]-ver2[0]*ver1[2])/(ver2[0]*ver1[1]-ver1[0]*ver2[1])
    n = np.array([A,B,1])/np.linalg.norm([A,B,1])
    return n,np.dot(points[0,:],n)

def index01(typ,index):
    if typ == 'index':
        return 0
    if typ == 'array01':
        index0 = []
        index1 = []
        for i in range(len(index)):
            if index[i]:
                index1.append(i)
            else:
                index0.append(i)
        return index0,index1
    
def partition(arr, low, high, sortkey):       #参数：列表，列表的第一个索引0，最后一个索引值N
    i = low                                       # 最小元素索引
    pivot = arr[high][sortkey]                             # 最后一个元素，我们把列表中的所有元素同它比较

    for j in range(low, high):                    #从第一个索引到倒数第二个索引
        if arr[j][sortkey] <= pivot:                       #从第一个元素到倒数第二个元素依次判断是否≤最后一个元素
            arr[i], arr[j] = arr[j], arr[i]       #≤最后一个元素的所有元素依次放在左边索引0~i的位置
            i = i + 1
    arr[i], arr[high] = arr[high], arr[i]         #然后将最后一个元素放在索引i的位置，实现：该元素左边的都比它小，右边的都比它大的排序
    return (i)                                    #返回该元素的索引位置

def QSort(arr, low, high, sortkey=0):
    if low < high:                                #如果列表有1个以上的元素
        pi = partition(arr, low, high, sortkey)            #获取左小右大函数中的 被比较数所在的索引

        QSort(arr, low, pi - 1, sortkey)            #反复循环，左排序
        QSort(arr, pi + 1, high, sortkey)           #反复循环，右排序


def groud_norm(pcd, pc_type = 'leica', visual = True, addr = '', write_points = 0):
    pcd_groud = copy.deepcopy(pcd)
    pc_groud = np.asarray(pcd_groud.points)
    pca = PCA(n_components=3)
    n = np.array([0,0,1])
    rate = 0
    for i in range(10):
        d = np.dot(pc_groud,n.T)
        D = np.quantile(d,0.25)
        array01 = d - D < 0.05###可调整阈值
        index0,index1 = index01('array01',array01)
        pca_points = pc_groud[index1]
        pca.fit(pca_points)
        n = pca.components_[2,:]
        if n[2] < 0:
            n = -n
        if len(index1)/len(pc_groud) - rate <= 0.001:
            break
        rate = len(index1)/len(pc_groud)
        print(n,rate)
    
    
    print("Groud points:%.2f",len(index1)/len(pc_groud))
    
    ## norm z dist and x axis
    mat1 = pt.generate_rotation_mat(0,math.atan(n[1]/n[2]))
    mat2 = pt.generate_rotation_mat(1,math.atan(n[0]/math.sqrt(n[2]*n[2]+n[1]*n[1])))
    mat = np.dot(mat2,mat1)
    xrot = np.dot(mat,np.array([10,0,0,1]))
    mat3 = pt.generate_rotation_mat(2,-math.atan(xrot[0,1]/xrot[0,0]))### x axis
    mat = np.dot(mat3,mat)
    pcd_groud_t = copy.deepcopy(pcd)
    pcd_groud_t.transform(mat)
    pc_z = np.asarray(pcd_groud_t.points)[index1,2]### z dist
    mat[2,3] = mat[2,3]-np.quantile(pc_z,0.25)
    pcd_groud.transform(mat)
    if visual:
        pt.display_inlier_outlier(pcd_groud, index1, change_color = True)
    if write_points >= 1:
        o3d.io.write_point_cloud(addr + '_gd.ply',pcd_groud)
    if write_points >= 2:
        pc_nogroud = np.asarray(pcd_groud.points)[index0]
        
        array01 = pc_nogroud[:,2] > 5###可调整阈值
        index0,index1 = index01('array01',array01)
        pc_nogroud = pc_nogroud[index0]
        
        pcd_groud.points = o3d.utility.Vector3dVector(pc_nogroud)
        o3d.io.write_point_cloud(addr + '_nogd.ply',pcd_groud)
    return mat,pc_z

    
if __name__ == '__main__':
    pc_type = ''
    ### Livox ###
    if pc_type == 'livox':
        pcd = o3d.io.read_point_cloud("./cross/leica.pts")
        # for i in range(50):
        #     source = o3d.io.read_point_cloud("./xqlk/NE_sync/"+str(i)+".pcd")
        #     pcd = pcd + source
    
    ### Leica ###
    if pc_type == 'leica':
        pcd = o3d.io.read_point_cloud("./cross/leica.pts")
        
    pcd = o3d.geometry.PointCloud()
    pcd = o3d.io.read_point_cloud("./data/20210808/NorthWest.pcd")
    mat,pcz = groud_norm(pcd, pc_type = pc_type, addr="./data/20210808",write_points = 2)
    # t = time.time()
    # pcd_list = []
    # for i in range(1):
    #     pcd = o3d.geometry.PointCloud()
    #     for j in range(50):
    #         source = o3d.io.read_point_cloud("./xqlk/NE_sync/"+str(j+i*50)+".pcd") 
    #         pcd = pcd + source
    #     pcd_list.append(pcd)
    # mat = groud_norm(pcd_list[0], pc_type = 'livox')
    # for i in range(1):
    #     pcd_list[i].transform(mat)
    #     pc_trans = np.asarray(pcd_list[i].points)
    #     pc_sort = pc_trans[:80000]
    #     print(time.time()-t)
    #     t = time.time()
    #     QSort(pc_sort.tolist(),0,len(pc_sort)-1,0)

