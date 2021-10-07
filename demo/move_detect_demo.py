import sys
sys.path.append(".")
import open3d as o3d
import matplotlib.pyplot as plt
from preprocess.livoxreader import LivoxReader
from tools.move_detect import compare_points, combine_inlier_outlier


if __name__ == '__main__':
    plt.ion()
    plt.figure(1)
    score_list = []
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    reader = LivoxReader(r"C:\Users\win10\Desktop\data\20211005\move2.lvx")
    ret, prev_pcd = reader.read_frame(num=20, voxel_size=0.01)
    while True:
        ret, curr_pcd = reader.read_frame(num=20, voxel_size=0.01)
        if not ret:
            break
        
        inlier, outlier = compare_points(prev_pcd, curr_pcd, distance_threshold=0.5)
        score = len(inlier) / (len(inlier) + len(outlier))
        score_list.append(score)
        plt.plot(range(1, len(score_list)+1), score_list, c='r',ls='-', marker='o', mec='b',mfc='w')
        plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        plt.ylabel("Similarity score")
        plt.pause(0.01)
        
        vis.clear_geometries()
        vis.add_geometry(combine_inlier_outlier(prev_pcd, inlier))
        vis.poll_events()
        vis.update_renderer()
        
        prev_pcd = curr_pcd
        
    vis.destroy_window()