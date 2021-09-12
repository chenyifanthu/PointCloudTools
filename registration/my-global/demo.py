import open3d as o3d
import numpy as np
from sklearn.neighbors import NearestNeighbors

def registration_ransac_based_on_feature_matching(source, target, source_fpfh, target_fpfh):
    source_fpfh = source_fpfh.data.T
    target_fpfh = target_fpfh.data.T
    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)
    search_model = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
    search_model.fit(source_fpfh)
    
    
    
if __name__ == '__main__':
    npoints =100
    h_threshold = 0.1
    n_neighbors = 5
    source_points = np.random.randn(npoints, 3)
    target_points = np.random.randn(npoints, 3)
    source_fpfh = np.random.randn(npoints, 32)
    target_fpfh = np.random.randn(npoints, 32)
    
    search_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree')
    search_model.fit(source_fpfh)
    
    ind = search_model.kneighbors(target_fpfh, return_distance=False)
    for k in range(n_neighbors):
    
        select_ind = ind[:, k]

        z_src = source_points[select_ind, 2]
        z_tar = target_points[:, 2]
        
        
        
        hvalid_ind = np.abs(z_src-z_tar) < h_threshold
        corr1 = ind[hvalid_ind]
        corr2 = np.array(range(target_points.shape[0]))[hvalid_ind]
        print(corr1, corr2)
        