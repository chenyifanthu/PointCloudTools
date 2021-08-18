import open3d as o3d
import pye57
import numpy as np


def read_leica_data(e57_file):
    print('Reading file: %s' % e57_file)
    e57 = pye57.E57(e57_file)
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
    
    print('Processing data...')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.array([data['X'], data['Y'], data['Z']]).T
    )
    pcd.colors = o3d.utility.Vector3dVector(
        np.array([data['colorRed'], data['colorGreen'], data['colorBlue']]).T / 255
    )
    
    return pcd

if __name__ == '__main__':
    pcd = read_leica_data('data/20210808.e57')
    pcd_down = pcd.voxel_down_sample(0.3)
    print(pcd_down)
    o3d.visualization.draw_geometries([pcd_down])