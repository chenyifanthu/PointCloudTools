import open3d as o3d

trans_mat1 = \
[[ 9.04138196e-01,  4.26846791e-01, -1.83286368e-02, -9.67771486e+00],
 [-4.27038684e-01,  9.04197135e-01, -8.09335790e-03,  5.14515155e+00],
 [ 1.31180770e-02,  1.51445510e-02,  9.99799259e-01,  3.07366140e-01],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]

trans_mat2 = \
[[ 7.86412169e-01, -6.17678584e-01, -5.39141104e-03, -8.22768994e+00],
 [ 6.17624035e-01,  7.86421585e-01, -9.03564178e-03, -5.95288902e+00],
 [ 9.82104444e-03,  3.77587361e-03,  9.99944643e-01,  3.01386255e-01],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]

trans_mat3 = \
[[-8.78415755e-01,  4.77858249e-01, -6.10360933e-03,  1.43896450e+01],
 [-4.77897228e-01, -8.78344324e-01,  1.12021724e-02,  7.54774877e+00],
 [-8.02013609e-06,  1.27570627e-02,  9.99918625e-01,  3.53819999e-01],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]

# trans_mat4 = 


if __name__ == '__main__':
    livox_1 = o3d.io.read_point_cloud('data/volleyball/livox_1.pcd')
    livox_1.transform(trans_mat1)
    livox_2 = o3d.io.read_point_cloud('data/volleyball/livox_2.pcd')
    livox_2.transform(trans_mat2)
    livox_3 = o3d.io.read_point_cloud('data/volleyball/livox_3.pcd')
    livox_3.transform(trans_mat3)
    # livox_1 = o3d.io.read_point_cloud('data/volleyball/livox_1.pcd')
    o3d.visualization.draw_geometries([livox_1, livox_3])