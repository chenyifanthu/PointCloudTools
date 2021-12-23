import cv2
import math
import time
import numpy as np
import open3d as o3d

CAM_EXTRINSIC_LIST = []

def decomposeT(T):
    return T[:3, :3], T[:3, 3]

def composeT(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T

def get_theta_rotaxis_from_rotmat(R):
    theta = math.acos((np.trace(R)-1)/2)
    w, v = np.linalg.eig(R)
    idx = np.argmax(np.real(w))
    n = np.real(v[:, idx])
    # print(w, idx)
    return theta, n

def get_rotmat_from_theta_rotaxis(theta, n):
    n_ = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]]).T
    n = n.reshape(-1, 1)
    return math.cos(theta) * np.eye(3) + (1 - math.cos(theta)) * n @ n.T + math.sin(theta) * n_

def check_valid(theta, n, R):
    R_ = get_rotmat_from_theta_rotaxis(theta, n)
    if not np.array_equal(R_, R):
        n = -n
    return theta, n

def add_camara_extrinsic(vis):
    vctr = vis.get_view_control()
    param = vctr.convert_to_pinhole_camera_parameters()
    CAM_EXTRINSIC_LIST.append(param.extrinsic)
    print('已添加第%d个视点' % len(CAM_EXTRINSIC_LIST))
    
def combine_extrinsic_intrinsic(extrinsic, intrinsic):
    param = o3d.camera.PinholeCameraParameters()
    param.extrinsic = extrinsic
    param.intrinsic = intrinsic
    return param
    
def generate_trajectory_from_extrinsic_list(ext_list, intrinsic, interval=100):
    param_init = combine_extrinsic_intrinsic(ext_list[0], intrinsic)
    param_list = [param_init]
    for i in range(1, len(ext_list)):
        T0, T1 = ext_list[i-1:i+1]
        R0, t0 = decomposeT(T0)
        R1, t1 = decomposeT(T1)
        R = R1 @ np.linalg.inv(R0)
        t = t1 - t0
        theta, n = get_theta_rotaxis_from_rotmat(R)
        # theta, n = check_valid(theta, n, R)
        # R_ = get_rotmat_from_theta_rotaxis(theta, n)
        # print(R, '\n', R_)
        for j in range(1, interval+1):
            R_ = get_rotmat_from_theta_rotaxis(j * theta / interval, n)
            t_ = j * t / interval
            T_ = composeT(R_ @ R0, t0 + t_)
            param = combine_extrinsic_intrinsic(T_, intrinsic)
            param_list.append(param)
        print(T_)
        print(T1)
        print('===============')
    trajectory = o3d.camera.PinholeCameraTrajectory()
    trajectory.parameters = param_list
    return trajectory
    
# def generate_image_sequence_from_trajectory(vis, trajectory, dir='./image'):
#     if not os.path.exists(dir):
#         os.makedirs(dir)
#     ctr = vis.get_view_control()
#     for idx, param in enumerate(trajectory.parameters):
#         ctr.convert_from_pinhole_camera_parameters(param)
#         vis.update_renderer()
#         time.sleep(0.05)


def custom_draw_geometry_with_camera_trajectory(pcd, trajectory):
    custom_draw_geometry_with_camera_trajectory.index = -1
    custom_draw_geometry_with_camera_trajectory.trajectory =trajectory
    custom_draw_geometry_with_camera_trajectory.vis = o3d.visualization.Visualizer()
    custom_draw_geometry_with_camera_trajectory.writer = cv2.VideoWriter(
        filename='./demo.mp4', fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=24, frameSize=(987, 1680)
    )
    
 
    def move_forward(vis):
        ctr = vis.get_view_control()
        glb = custom_draw_geometry_with_camera_trajectory
        glb.index = glb.index + 1
        if glb.index < len(glb.trajectory.parameters):
            ctr.convert_from_pinhole_camera_parameters(glb.trajectory.parameters[glb.index])
            image = vis.capture_screen_float_buffer(False)
            image = np.array(image)
            glb.writer.write(image)
        else:
            custom_draw_geometry_with_camera_trajectory.vis.register_animation_callback(None)
        # time.sleep(0.04)
        return False
 
    vis = custom_draw_geometry_with_camera_trajectory.vis
    vis.create_window(width=1680, height=987, left=0, top=0)
    vis.add_geometry(pcd)
    vis.register_animation_callback(move_forward)
    vis.run()
    vis.destroy_window()
    custom_draw_geometry_with_camera_trajectory.writer.release()
    
    
if __name__ == '__main__':
    height = 987
    width = 1680
    # fov = math.radians(60)

    fx = fy = math.sqrt(3) / 2 * height
    cx, cy = (width-1)/2, (height-1)/2
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    
    pcd = o3d.io.read_point_cloud('leica.ply')
    
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=width, height=height, left=0, top=0)
    vis.register_key_callback(ord("A"), add_camara_extrinsic)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    
    # o3d.io.write_pinhole_camera_parameters('param1.json', combine_extrinsic_intrinsic(CAM_EXTRINSIC_LIST[0], camera_intrinsic))
    # o3d.io.write_pinhole_camera_parameters('param2.json', combine_extrinsic_intrinsic(CAM_EXTRINSIC_LIST[1], camera_intrinsic))

    
    trajectory = generate_trajectory_from_extrinsic_list(CAM_EXTRINSIC_LIST, camera_intrinsic)
    o3d.io.write_pinhole_camera_trajectory('trajectory.json', trajectory)
    
    # trajectory = o3d.io.read_pinhole_camera_trajectory('trajectory.json')
    custom_draw_geometry_with_camera_trajectory(pcd, trajectory)