import cv2
import math
import time
import numpy as np
import open3d as o3d


def decomposeT(T):
    return T[:3, :3], T[:3, 3]

def composeT(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T

def get_rotmat_from_quaternion(q):
    return o3d.geometry.get_rotation_matrix_from_quaternion(q).T

def get_theta_n_from_quaternion(q):
    theta = 2*math.acos(q[0])
    n = q[1:]/math.sin(theta/2)
    return theta, n

def get_quaternion_from_theta_n(theta, n):
    return np.concatenate(([math.cos(theta/2)], n * math.sin(theta/2)))

def get_quaternion_from_rotmat(R):
    q0 = math.sqrt(np.trace(R)+1)/2
    q1 = (R[1,2]-R[2,1])/q0/4
    q2 = (R[2,0]-R[0,2])/q0/4
    q3 = (R[0,1]-R[1,0])/q0/4
    return np.array([q0,q1,q2,q3])

def save_camera_extrinsic_interactive(pcd, window_size=(1920, 1080)):
    
    cam_extrinsic_list = []
    def add_camera_extrinsic(vis):
        vctr = vis.get_view_control()
        param = vctr.convert_to_pinhole_camera_parameters()
        cam_extrinsic_list.append(param.extrinsic)
        print('已添加第%d个视点:' % len(cam_extrinsic_list))
        print(param.extrinsic)
        
    vis = o3d.visualization.VisualizerWithKeyCallback()
    width, height = window_size
    vis.create_window(width=width, height=height, left=0, top=0)
    vis.register_key_callback(ord("A"), add_camera_extrinsic)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    return cam_extrinsic_list

def interpolate_extrinsic_matrix(ext_list, interval=50):
    ext_list_new = [ext_list[0]]
    for i in range(1, len(ext_list)):
        T0, T1 = ext_list[i-1:i+1]
        R0, t0 = decomposeT(T0)
        R1, t1 = decomposeT(T1)
        R = R1 @ np.linalg.inv(R0)
        t = t1 - t0
        q = get_quaternion_from_rotmat(R)
        theta, n = get_theta_n_from_quaternion(q)
        for j in range(1, interval+1):
            q_ = get_quaternion_from_theta_n(j * theta / interval, n)
            R_ = get_rotmat_from_quaternion(q_)
            t_ = j * t / interval
            T_ = composeT(R_ @ R0, t0 + t_)
            ext_list_new.append(T_)
    return ext_list_new

def combine_extrinsic_intrinsic(extrinsic, intrinsic):
    param = o3d.camera.PinholeCameraParameters()
    param.extrinsic = extrinsic
    param.intrinsic = intrinsic
    return param

def generate_trajectory(ext_list, intrinsic):
    param_list = [combine_extrinsic_intrinsic(extrinsic, intrinsic) for extrinsic in ext_list]
    trajectory = o3d.camera.PinholeCameraTrajectory()
    trajectory.parameters = param_list
    return trajectory

def custom_draw_geometry_with_camera_trajectory(pcd, trajectory, filename, window_size=(1920, 1080)):
    custom_draw_geometry_with_camera_trajectory.index = -1
    custom_draw_geometry_with_camera_trajectory.trajectory = trajectory
    custom_draw_geometry_with_camera_trajectory.vis = o3d.visualization.Visualizer()
    custom_draw_geometry_with_camera_trajectory.writer = cv2.VideoWriter(filename, 
                                                                         cv2.VideoWriter_fourcc(*'mp4v'), 
                                                                         30, window_size)
    
    def move_forward(vis):
        ctr = vis.get_view_control()
        glb = custom_draw_geometry_with_camera_trajectory
        glb.index = glb.index + 1
        if glb.index < len(glb.trajectory.parameters):
            ctr.convert_from_pinhole_camera_parameters(glb.trajectory.parameters[glb.index])
            image = vis.capture_screen_float_buffer(False)
            image_arr = (255*np.asarray(image)).astype(np.uint8)
            image_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB)
            glb.writer.write(image_arr)
        else:
            vis.destroy_window()
            glb.writer.release()
        return False
    
    print('Generating demo vedio...')
    w, h = window_size
    vis = custom_draw_geometry_with_camera_trajectory.vis
    vis.create_window(width=w, height=h, left=0, top=0, visible=False)
    opt = vis.get_render_option()
    opt.point_size = 2.0
    vis.add_geometry(pcd)
    vis.register_animation_callback(move_forward)
    vis.run()

def generate_circle_trajectory(center, radius, interval=50):
    ext_list = []
    for i in range(interval):
        theta = 2 * i * math.pi / interval
        R = np.array([[math.cos(theta), math.sin(theta), 0],
                    [0, 0, -1],
                    [-math.sin(theta), math.cos(theta), 0]])
        t = - R @ np.array(center) + np.array([0, 0, radius])
        T = composeT(R, t)
        ext_list.append(T)
    return ext_list
    
def pick_points(pcd):
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    idx = vis.get_picked_points()[0]
    return pcd.points[idx]


if __name__ == '__main__':
    height = 987
    width = 1680
    
    fx = fy = math.sqrt(3) / 2 * height
    cx, cy = (width-1)/2, (height-1)/2
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    
    pcd = o3d.io.read_point_cloud(r"D:\data\20210918\leica.ply")
    
    center = pick_points(pcd)
    # center = [-10, -0.47, 0.63]
    ext_list = generate_circle_trajectory(center, radius=5, interval=300)
    trajectory = generate_trajectory(ext_list, intrinsic)
    custom_draw_geometry_with_camera_trajectory(pcd, trajectory, 'demo-center.mp4', (width, height))
    
    
    # ext_list = save_camera_extrinsic_interactive(pcd, (width, height))
    # np.save('ext_data.npy', np.array(ext_list))
    # # ext_list = list(np.load('ext_data.npy')[:2])

    # ext_list_interp = interpolate_extrinsic_matrix(ext_list, interval=120)
    # trajectory = generate_trajectory(ext_list_interp, intrinsic)
    # custom_draw_geometry_with_camera_trajectory(pcd, trajectory, 'demo-line.mp4', (width, height))