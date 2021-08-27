import open3d as o3d


if __name__ == '__main__':
    pcd = o3d.io.read_point_cloud('data/20210808/leica_down.ply')
    o3d.visualization.draw_geometries([pcd])

    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=30))

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=100))
    
    print(fpfh.data.T)

