import numpy as np
from math import sin, cos, atan, acos

def normalize(vec):
    return vec / np.linalg.norm(vec)
    
# def get_rotation_matrix(x_rot, y_rot, z_rot):
#     rx = np.array([[1, 0, 0], 
#                    [0, cos(x_rot), -sin(x_rot)], 
#                    [0, sin(x_rot), cos(x_rot)]])
#     ry = np.array([[cos(y_rot), 0, sin(y_rot)], 
#                    [0, 1, 0],
#                    [-sin(y_rot), 0, cos(y_rot)]])
#     rz = np.array([[cos(z_rot), -sin(z_rot), 0], 
#                    [sin(z_rot), cos(z_rot), 0],
#                    [0, 0, 1]])
#     return rx.dot(ry).dot(rz)
    
def get_transform_matrix_from_plane_function(coef):
    assert len(coef) == 4
    a, b, c, d = coef
    z_axis = normalize(np.array([a, b, c]))
    x_axis = normalize(np.array([b, -a, 0]))
    y_axis = np.cross(z_axis, x_axis)
    R = np.vstack((x_axis, y_axis, z_axis))
    t = np.array([[0, 0, d*z_axis[0]/a]])
    T = np.block([[R, t.T], [np.zeros((1, 3)), 1]])
    return T
    
if __name__ == '__main__':
    pass