import open3d as o3d
import numpy as np
import copy
# import math 
import pctool as ptl
import pdb
import time
#点云绘制函数
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([source_temp.transform(transformation), target_temp])
    

#数据输入函数(降采样预处理)
def prepare_dataset(source,target,trans_init,voxel_size):
    print("prepare_dataset strat")
    source.transform(trans_init)
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

#提取几何特征
def preprocess_point_cloud(pcd, voxel_size):  # 传入参数pcd点云数据，voxel_size体素大小
    print("preprocess_point_cloud strat")
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)  # 降采样

    radius_normal = voxel_size * 2  # kdtree参数，用于估计法线的半径，一般设为体素大小的2倍
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))  # 估计法线的1个参数，使用混合型的kdtree，半径内取最多30个邻居

    radius_feature = voxel_size * 5  # kdtree参数，用于估计特征的半径，设为体素大小的5倍
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))  # 计算特征的2个参数，下采样的点云数据，搜索方法kdtree
    return pcd_down, pcd_fpfh  # 返回降采样的点云、fpfh特征

def execute_global_registration(source_down, target_down, source_fpfh,target_fpfh, voxel_size):  # 传入2的点云的降采样，两个点云的特征、体素大小
    print("execute_global_registration strat")
    distance_threshold = voxel_size * 1.5  # 设定距离阈值为体素的1.5倍
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    # 2个降采样的点云，两个点云的特征，距离阈值，一个函数，4，一个list[0.9的两个对应点的线段长度阈值，两个点的距离阈值]，一个函数设定最大迭代次数和最大验证次数
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    return result

def execute_fast_global_registration(source_down, target_down, source_fpfh,target_fpfh, voxel_size):
    print("execute_fast_global_registration strat")
    distance_threshold = voxel_size * 0.5
    # print(":: Apply fast global registration with distance threshold %.3f" \
    #         % distance_threshold)
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

def ICP(target_ori,source_ori,dist=[0,0,0],rot=[0,0,0],visual = True,ICP = False,print_mat = False,global_registration = 'no'):
    pre_trans = np.asarray([[1,0,0,dist[0]],[0,1,0,dist[1]],[0,0,1,dist[2]],[0,0,0,1]])
    pre_trans = np.dot(ptl.generate_rotation_mat(2,rot[2]/180*3.1415926),pre_trans)
    pre_trans = np.dot(ptl.generate_rotation_mat(1,rot[1]/180*3.1415926),pre_trans)
    pre_trans = np.dot(ptl.generate_rotation_mat(0,rot[0]/180*3.1415926),pre_trans)
    size = 1
    pre_trans = np.dot(np.asarray([[size,0,0,0], 
                            [0,size,0,0],
                            [0,0,size,0], 
                            [0,0,0,1]]),pre_trans)
    # pre_trans = [[-6.00070672e-01, -7.99497307e-01, -2.68187236e-02,
    #      1.00410044e+01],
    #    [ 7.99926255e-01, -5.99959194e-01, -1.29210363e-02,
    #     -1.10852170e+01],
    #    [-5.75980607e-03, -2.92065361e-02,  9.99556804e-01,
    #      9.67696750e-02],
    #    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    #      1.00000000e+00]]
    if visual:
        draw_registration_result(source_ori, target_ori, pre_trans)
    source = copy.deepcopy(source_ori)
    target = copy.deepcopy(target_ori)
    if ICP:    
        voxel_size = 1  # 体素大小
        source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source,target,pre_trans,voxel_size)
        
        ###是否进行全局匹配
        global_reg_trans = np.eye(4)
        ###快速全局特征配准
        if global_registration == 'fast':
            draw_registration_result(source_down, target_down,np.eye(4)) 
            result_fast = execute_fast_global_registration(source_down, target_down,source_fpfh, target_fpfh,voxel_size)
            # print(result_fast)
            draw_registration_result(source_down, target_down,result_fast.transformation)
            global_reg_trans = result_fast.transformation
        ###全局特征配准
        if global_registration == 'normal':
            draw_registration_result(source_down, target_down,np.eye(4)) 
            result_ransac = execute_global_registration(source_down, target_down,source_fpfh, target_fpfh,voxel_size)
            # print(result_ransac)
            draw_registration_result(source_down, target_down, result_ransac.transformation)
            global_reg_trans = result_ransac.transformation
            
        ###局部点云ICP
        # reg_p2l = refine_registration(source, target, source_fpfh, target_fpfh, voxel_size)
        
        
        threshold = voxel_size * 0.4 
        reg_p2l = o3d.pipelines.registration.registration_icp(
            source, target_down, threshold, global_reg_trans,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        
        if visual:
            draw_registration_result(source, target, reg_p2l.transformation)
        
        if print_mat:
            transformation = np.dot(reg_p2l.transformation,pre_trans)
            print(transformation)
        return reg_p2l,transformation

if __name__ == '__main__':
    target = o3d.io.read_point_cloud('D:/StudyFile/keyan/pointcloud/cross/leica_nogd.ply')
    source = o3d.io.read_point_cloud('D:/StudyFile/keyan/pointcloud/cross/NorthWest_nogd.ply')
    t = time.time()
    
    dist = [-15,-2,0]
    rot = [0,0,215]
    reg_p2l,mat = ICP(target,source,dist,rot,visual=True,ICP = False,print_mat=True)
    print(reg_p2l)
    
    # max_fit = 0
    # for i in range(10):
    #     for j in range(20):
    #         dist0 = -20 - (i-5)
    #         rot2 = 330 - (j-10)*2
    #         dist = [dist0,0,0]
    #         rot = [0,0,rot2]
    #         reg_p2l = ICP(target,source,dist,rot,visual=False,ICP = True)
    #         print(reg_p2l.fitness,dist[0],rot[2],'time:',time.time()-t)
    #         if reg_p2l.fitness > max_fit:
    #             max_fit = reg_p2l.fitness
    #             max_xz = [dist0,rot2]
        
        

