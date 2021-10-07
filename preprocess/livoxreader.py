import os
import struct
import numpy as np
import open3d as o3d

class LivoxReader:
    def __init__(self, filepath: str) -> None:
        self.path = filepath
        self.filename = os.path.basename(filepath).split('.')[0]
        self.filetype = os.path.basename(filepath).split('.')[1]
        assert self.filetype == "lvx"
        
        self.MIN_POINT_NUM = 1000
        self.DATA = open(filepath,"rb").read()
        self.bytes_counter = 0
        self.read_header()
        
    def read_header(self):
        PRIVATE_HEADER_BLOCK = self.DATA[:5]
        self.bytes_counter += 5
        FRAME_DURATION, DEVICE_COUNT = struct.unpack("=IB", PRIVATE_HEADER_BLOCK)
        self.bytes_counter += DEVICE_COUNT * 59
        
    def read_frame(self, num=1, voxel_size=0.01):
        count = 0
        points = []
        while count < num:
            frame_header = self.DATA[self.bytes_counter:self.bytes_counter+24]
            if len(frame_header) != 24:
                return False, None
            self.bytes_counter += 24
            frame_points = []
            while True:
                PACKAGE_BEGIN_BLOCK = self.DATA[self.bytes_counter:self.bytes_counter+19]
                if len(PACKAGE_BEGIN_BLOCK) != 19:
                    break
                device_idx, version, slot_id, lidar_id, reserved, status_code, timestamp_type, data_type, timestamp = struct.unpack(
                    "<BBBBBIBBQ", PACKAGE_BEGIN_BLOCK)
                if version != 5:
                    break
                self.bytes_counter += 19

                if data_type == 0:
                    for _ in range(100):
                        point_block = self.DATA[self.bytes_counter:self.bytes_counter+13]
                        x,y,z,r = struct.unpack("<iiiB",point_block)
                        frame_points.append([x,y,z,r])
                        self.bytes_counter += 13
                        
            if len(frame_points) > self.MIN_POINT_NUM:
                points.extend(frame_points)
                count += 1
                    
        points = np.array(points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3] / 1000)
        if voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size)
        return True, pcd
    



if __name__ == '__main__':
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    reader = LivoxReader(r"C:\Users\win10\Desktop\data\20211005\move2.lvx")
    
    while True:
        # 读取接下来的20帧点云
        ret, frame_pcd = reader.read_frame(20)
        if not ret:
            break
        # frame_pcd = frame_pcd.voxel_down_sample(0.01)

        vis.clear_geometries()
        vis.add_geometry(frame_pcd)
        vis.poll_events()
        vis.update_renderer()

    vis.destroy_window()
