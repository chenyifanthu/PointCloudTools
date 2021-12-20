import glob
import numpy as np
import open3d as o3d
from tqdm import tqdm
import matplotlib.pyplot as plt
from registration.time_calibration import *

## Settings
voxel_size = 0.1
max_distance = 20
addn = 40
ref_frame_id = 7000
frame_start_id = 7000
frame_end_id = 7100
vis = True
ref_dir = r'E:\data\4D_CT_20211202\cam3\tf_pointcloud\*.pcd'
search_dir = r'E:\data\4D_CT_20211202\cam4\tf_pointcloud\*.pcd'


ref_filelist = glob.glob(ref_dir)
ref_reader = LivoxStream(ref_filelist, addn, ref_frame_id)
ref_frame = ref_reader.get_pcd(voxel_size, max_distance, color=[1, 0, 0])
ref_tree = setup_kdtree(ref_frame)

filelist = glob.glob(search_dir)
reader = LivoxStream(filelist, addn, frame_start_id)


dist_list = []
t_list = []

if vis:
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    plt.figure(figsize=(6,4))
    plt.ion()
    
for t in tqdm(range(frame_start_id, frame_end_id)):
    frame = reader.get_pcd(voxel_size, max_distance, color=[0, 0, 1])
    t_list.append(t)
    dist_list.append(distance_between_pc(frame, ref_tree, 1))
    reader.next()
    
    if vis:
        plt.clf()
        plt.plot(t_list, dist_list, color='blue')
        plt.xlim(frame_start_id, frame_end_id)
        plt.tight_layout()
        plt.pause(0.1)
        plt.ioff()
        
        vis.clear_geometries()
        vis.add_geometry(ref_frame)
        vis.add_geometry(frame)
        vis.poll_events()
        vis.update_renderer()

offset = frame_start_id + np.argmin(dist_list) - ref_frame_id
print("Offset:", offset)