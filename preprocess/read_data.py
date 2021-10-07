import os
import glob
import struct
import pye57
import numpy as np
import open3d as o3d
from tqdm import tqdm

SUPPORT_SUFFIX = ['.e57', '.lvx', '.pcd', '.ply', '.xyz']


def read_leica_data(filepath):
    e57 = pye57.E57(filepath)
    data = e57.read_scan_raw(0)
    '''
    data is a dictionary with keys:
        - sphericalRange: 点距离
        - sphericalAzimuth: 方位角(from 0 to 2pi)
        - sphericalElevation: 俯仰角(from -pi/2 to pi)
        - intensity: 点云强度
        - colorRed: 点云R通道颜色
        - colorGreen: 点云G通道颜色
        - colorBlue: 点云B通道颜色
        - rowIndex
        - columnIndex
        - sphericalInvalidState
    '''
    data['X'] = data['sphericalRange'] * np.cos(data['sphericalElevation']) * np.cos(data['sphericalAzimuth'])
    data['Y'] = data['sphericalRange'] * np.cos(data['sphericalElevation']) * np.sin(data['sphericalAzimuth'])
    data['Z'] = data['sphericalRange'] * np.sin(data['sphericalElevation'])
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.array([data['X'], data['Y'], data['Z']]).T
    )
    pcd.colors = o3d.utility.Vector3dVector(
        np.array([data['colorRed'], data['colorGreen'], data['colorBlue']]).T / 255
    )
    
    return pcd


def read_livox_data(filepath):
    
    fd = open(filepath,"rb")
    bytes_counter = 0
    DATA = fd.read()

    PRIVATE_HEADER_BLOCK = DATA[:5]
    bytes_counter += 5
    FRAME_DURATION, DEVICE_COUNT = struct.unpack("=IB", PRIVATE_HEADER_BLOCK)

    # print(f"FRAME_DURATION is {FRAME_DURATION}")
    # print(f"DEVICE COUNT is {DEVICE_COUNT}")

    devices_info = []
    for _ in range(DEVICE_COUNT):
        device_info_block = DATA[bytes_counter:bytes_counter+59]
        bytes_counter += 59

        LIDAR_SN, HUB_SN, DEVICE_IDX, DEVICE_TYPE, EXTRINSIC_ENABLE, ROLL, PITCH, YAW, X, Y, Z = struct.unpack(
            "=16s16sBBBffffff", device_info_block)
        devices_info.append([LIDAR_SN, HUB_SN, DEVICE_IDX, DEVICE_TYPE, EXTRINSIC_ENABLE, ROLL, PITCH, YAW, X, Y, Z])

    points = []

    while True:
        frame_header = DATA[bytes_counter:bytes_counter+24]
        if len(frame_header) != 24:
            break
        # current_offset, next_offset, frame_index = struct.unpack("<qqq",frame_header )
        bytes_counter += 24

        frame_points = []
        while True:
            PACKAGE_BEGIN_BLOCK = DATA[bytes_counter:bytes_counter+19]

            if len(PACKAGE_BEGIN_BLOCK) != 19:
                points.extend(frame_points)
                break
            device_idx, version, slot_id, lidar_id, reserved, status_code, timestamp_type, data_type, timestamp = struct.unpack(
                "<BBBBBIBBQ", PACKAGE_BEGIN_BLOCK)

            if version != 5:
                points.extend(frame_points)
                break

            bytes_counter += 19

            if data_type == 0:
                #read 100 points * 13
                for _ in range(100):
                    point_block = DATA[bytes_counter: bytes_counter + 13]
                    if len(point_block) != 13:
                        points.extend(frame_points)
                        break
                    x,y,z,r = struct.unpack("<iiiB",point_block)
                    frame_points.append([x,y,z,r])
                    bytes_counter += 13
        

    fd.close()
    points = np.array(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3] / 1000)
    
    return pcd


def read_point_cloud(filepath):
    assert os.path.exists(filepath), "File %s doesn't exist" % filepath
    suffix = os.path.splitext(filepath)[1]
    assert suffix in SUPPORT_SUFFIX, "File type %s can't be recognized" % suffix
    
    if suffix == ".e57":
        return read_leica_data(filepath)
    elif suffix == '.lvx':
        return read_livox_data(filepath)
    else:
        return o3d.io.read_point_cloud(filepath)


def convert_single_point_cloud(filepath, outdir):
    filename, _ = os.path.splitext(os.path.basename(filepath))
    outfile = os.path.join(outdir, filename + ".ply")
    pcd = read_point_cloud(filepath)
    o3d.io.write_point_cloud(outfile, pcd)
    

def convert_point_cloud_in_directory(indir, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    filelist = glob.glob(os.path.join(indir, '*'))
    for file in tqdm(filelist):
        suffix = os.path.splitext(file)[1]
        if suffix in SUPPORT_SUFFIX:
            convert_single_point_cloud(file, outdir)


if __name__ == '__main__':
    convert_point_cloud_in_directory(r"C:\Users\win10\Desktop\DC", 
                                     r"C:\Users\win10\Desktop\data\20210923")